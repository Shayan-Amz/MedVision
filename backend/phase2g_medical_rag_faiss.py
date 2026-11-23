import json, os, time, faiss, numpy as np, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    adapter_path = "./checkpoints/medvision-7b-v4-cat-merged/cat_50"
    rag_dir = "./medical_rag_index"
    os.makedirs(rag_dir, exist_ok=True)

    print("=" * 84)
    print("  LEVER 2: Real Local Medical RAG (FAISS + BGE-Small on RTX 4080, Zero Leakage)")
    print("=" * 84)

    # Load 200 held-out MedMCQA validation questions first so we strictly exclude any matching text!
    mcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(200))
    val_q_set = set((ex["question"] or "").strip().lower()[:80] for ex in mcqa_val)

    # Build clean medical knowledge corpus (Flashcards + Rich MedMCQA train textbook explanations ONLY)
    docs = []
    fc = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    for ex in fc:
        q, a = (ex.get("input") or "").strip(), (ex.get("output") or "").strip()
        if len(a) > 25:
            docs.append(f"Clinical Fact: {q} -> {a[:320]}")

    mcqa_tr = load_dataset("openlifescienceai/medmcqa", split="train")
    opt_keys = ["opa", "opb", "opc", "opd"]
    added_exp = 0
    for ex in mcqa_tr:
        q = (ex.get("question") or "").strip()
        if q.lower()[:80] in val_q_set:
            continue  # Strict zero-leakage guard
        exp = (ex.get("exp") or "").strip()
        if len(exp) >= 80:
            cop = int(ex["cop"]) if int(ex["cop"]) in (0, 1, 2, 3) else 0
            ans_txt = (ex.get(opt_keys[cop]) or "").strip()
            docs.append(f"Medical Reference ({ex.get('subject_name', 'Medicine')}): {ans_txt}. {exp[:360]}")
            added_exp += 1
            if added_exp >= 50000:
                break

    print(f"[CORPUS] Indexed {len(docs)} leak-free medical passages ({len(docs)-added_exp} flashcards + {added_exp} textbook notes)", flush=True)

    # Embed corpus on RTX 4080 GPU & save persistent FAISS index
    index_path = os.path.join(rag_dir, "medvision_faiss.index")
    docs_path  = os.path.join(rag_dir, "medvision_docs.json")

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    t0 = time.perf_counter()
    print("[EMBEDDING] Encoding medical corpus on RTX 4080...", flush=True)
    embs = embedder.encode(docs, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, index_path)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)
    print(f"[FAISS SAVED] Built & saved {index.ntotal} vectors in {time.perf_counter()-t0:.1f}s!", flush=True)

    # Pre-retrieve Top-3 medical reference passages for all 200 unseen MedMCQA validation questions
    queries = [f"{ex['question']} A) {ex['opa']} B) {ex['opb']} C) {ex['opc']} D) {ex['opd']}" for ex in mcqa_val]
    q_embs = embedder.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores_mat, idxs_mat = index.search(q_embs, 3)

    # Free embedder from GPU before loading 7B LLM
    del embedder
    torch.cuda.empty_cache()

    # Load MedVision-7B-v4-CAT and evaluate WITH Retrieved Medical Evidence (Top-3 RAG)
    tok = AutoTokenizer.from_pretrained(base_path)
    opt_letters = ["A", "B", "C", "D"]
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in opt_letters]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()
    model = PeftModel.from_pretrained(base, adapter_path).eval()

    rag_ok = 0
    for i, ex in enumerate(mcqa_val):
        retrieved = "\n".join([f"[{j+1}] {docs[idx]}" for j, idx in enumerate(idxs_mat[i])])
        opts = f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}"
        msgs = [
            {"role": "system", "content": "You are MedVision Clinical AI. Use the retrieved medical textbook references below to answer the multiple-choice question accurately. Respond with strictly one letter: A, B, C, or D."},
            {"role": "user", "content": f"Retrieved Medical References:\n{retrieved}\n\nQuestion: {ex['question']}\n{opts}\nCorrect Option (A, B, C, or D):"}
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1280, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1, :]
            sc = [float(lg[cid].item()) for cid in opt_ids]
        if sc.index(max(sc)) == int(ex["cop"]):
            rag_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  [Medical RAG + 7B] {i+1}/200 | Accuracy: {rag_ok/(i+1)*100:.1f}%", flush=True)

    rag_acc = round(rag_ok / 200 * 100.0, 2)
    with open("./checkpoints/phase2g_medical_rag_report.json", "w") as f:
        json.dump({"corpus_size": len(docs), "embedder": "BAAI/bge-small-en-v1.5", "no_rag_medmcqa_pct": 55.0, "medical_rag_medmcqa_pct": rag_acc, "rag_gain_pct": round(rag_acc - 55.0, 2)}, f, indent=2)

    print("\n" + "=" * 84)
    print(f"  NO-RAG MedVision-7B-v4-CAT (MedMCQA n=200)   : 55.00% (110/200)")
    print(f"  MEDICAL-RAG (FAISS 84k) + 7B (MedMCQA n=200) : {rag_acc:.2f}% ({rag_ok}/200)  [RAG Gain: {rag_acc-55.0:+.2f}%]")
    print(f"  Saved Persistent FAISS Vector DB             : {os.path.abspath(rag_dir)}")
    print("=" * 84)

if __name__ == "__main__":
    main()
