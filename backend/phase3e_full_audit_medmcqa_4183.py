import gc, json, os, time, faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("=" * 90)
print("  PHASE 3E (Exact Quality + Zero-Spill): NIH-14 Audit + Full 4,183 MedMCQA Validation")
print("=" * 90)

with open("./nih_cxr14_cache/splits_meta.json", "r", encoding="utf-8") as f:
    nih_meta = json.load(f)
tr_files = {r["file"] for r in nih_meta["train"]}
va_files = {r["file"] for r in nih_meta["val"]}
te_files = {r["file"] for r in nih_meta["test"]}
print(f"[AUDIT NIH-14] Train: {len(tr_files)} | Val: {len(va_files)} | Test: {len(te_files)}")
print(f"  Overlap Train∩Val: {len(tr_files & va_files)} | Train∩Test: {len(tr_files & te_files)} | Val∩Test: {len(va_files & te_files)}")

ds_mcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
mcqa_val = [ex for ex in ds_mcqa if ex.get("cop") in (0, 1, 2, 3)]
print(f"\n[MedMCQA] Loaded {len(mcqa_val)} valid validation questions.")

rag_dir = "./medical_rag_index_full"
index = faiss.read_index(os.path.join(rag_dir, "full_faiss.index"))
with open(os.path.join(rag_dir, "full_docs.json"), "r", encoding="utf-8") as f:
    docs = json.load(f)

t0 = time.perf_counter()
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
queries = [f"{ex['question']} A) {ex['opa']} B) {ex['opb']} C) {ex['opc']} D) {ex['opd']}" for ex in mcqa_val]
q_embs = embedder.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
_, I = index.search(q_embs, 20)
del embedder; gc.collect(); torch.cuda.empty_cache()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
pairs = []
for i, ex in enumerate(mcqa_val):
    q_short = f"{ex['question']} ({ex['opa']}; {ex['opb']}; {ex['opc']}; {ex['opd']})"
    for idx in I[i]:
        pairs.append((q_short, docs[int(idx)]))

scores = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
rag_ctxs = []
for i in range(len(mcqa_val)):
    cand_idxs = I[i]
    cand_scores = scores[i * 20 : (i + 1) * 20]
    top3 = [docs[int(cand_idxs[j])] for j in np.argsort(cand_scores)[::-1][:3]]
    rag_ctxs.append("\n".join(f"- {c[:450]}" for c in top3))

del reranker, pairs, scores, q_embs, I, index, docs
gc.collect(); torch.cuda.empty_cache()
print(f"[RAG] Retrieved & reranked Top-3 evidence in {time.perf_counter()-t0:.1f}s.")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained("./qwen7b_local")
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "./qwen7b_local",
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/medvision-7b-v3-supermodel")
model.eval()

letters = ["A", "B", "C", "D"]
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]

correct_cyclic = 0
correct_single = 0
t1 = time.perf_counter()

with torch.inference_mode():
    for i, ex in enumerate(mcqa_val):
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        ctx = rag_ctxs[i]
        prompts = []
        for shift in range(4):
            rotated = [opts[(k + shift) % 4] for k in range(4)]
            user_msg = (
                f"Medical Reference Context:\n{ctx}\n\n"
                f"Question: {ex['question']}\n"
                f"A) {rotated[0]}\nB) {rotated[1]}\nC) {rotated[2]}\nD) {rotated[3]}\n"
                f"Answer with only the single letter (A, B, C, or D):"
            )
            txt = tok.apply_chat_template(
                [{"role": "system", "content": "You are an expert physician answering medical board exam questions accurately."},
                 {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True
            )
            prompts.append(txt)

        probs_list = []
        for s in (0, 2):
            enc = tok(prompts[s:s+2], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            logits = model(**enc, use_cache=False).logits[:, -1, :][:, letter_ids]
            probs_list.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.vstack(probs_list)

        if int(np.argmax(probs[0])) == int(ex["cop"]):
            correct_single += 1

        agg = np.zeros(4, dtype=np.float64)
        for shift in range(4):
            for k in range(4):
                agg[(k + shift) % 4] += probs[shift, k]
        if int(np.argmax(agg)) == int(ex["cop"]):
            correct_cyclic += 1

        if (i + 1) % 250 == 0 or (i + 1) == len(mcqa_val):
            dt = time.perf_counter() - t1
            eta = (dt / (i + 1)) * (len(mcqa_val) - (i + 1))
            print(f"  [{i+1:4d}/{len(mcqa_val)}] Single: {correct_single/(i+1)*100:5.2f}% | Cyclic-4 RAG: {correct_cyclic/(i+1)*100:5.2f}% | Elapsed: {dt/60:.1f}m | ETA: {eta/60:.1f}m", flush=True)

acc_single = round(correct_single / len(mcqa_val) * 100, 2)
acc_cyclic = round(correct_cyclic / len(mcqa_val) * 100, 2)
out_json = {
    "n_total": len(mcqa_val),
    "single_pass_rag_acc_pct": acc_single,
    "cyclic4_rag_acc_pct": acc_cyclic,
    "nih14_counts": {"train": len(tr_files), "val": len(va_files), "test": len(te_files)},
    "nih14_overlaps": {
        "train_val": len(tr_files & va_files),
        "train_test": len(tr_files & te_files),
        "val_test": len(va_files & te_files),
    },
}
with open("./checkpoints/phase3e_medmcqa_full4183_audit.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2)
print("\n[DONE] Saved -> ./checkpoints/phase3e_medmcqa_full4183_audit.json")
