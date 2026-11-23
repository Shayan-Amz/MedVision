import csv, gc, json, os, re, time, faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def norm_txt(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()

ds_mcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
mcqa_val = [ex for ex in ds_mcqa if ex.get("cop") in (0, 1, 2, 3)]
N_TOTAL = len(mcqa_val)
print(f"[MedMCQA] Loaded {N_TOTAL} validation questions.", flush=True)

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
for i in range(N_TOTAL):
    cand_idxs = I[i]
    cand_scores = scores[i * 20 : (i + 1) * 20]
    top3 = [docs[int(cand_idxs[j])] for j in np.argsort(cand_scores)[::-1][:3]]
    rag_ctxs.append("\n".join(f"- {c[:450]}" for c in top3))

del reranker, pairs, scores, q_embs, I, index, docs
gc.collect(); torch.cuda.empty_cache()
print(f"[RAG] Retrieved & reranked Top-3 evidence in {time.perf_counter()-t0:.1f}s.", flush=True)

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("./qwen7b_local")
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained("./qwen7b_local", quantization_config=bnb_cfg, device_map="auto", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "./checkpoints/medvision-7b-v3-supermodel").eval()

letters = ["A", "B", "C", "D"]
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
sys_msg = "You are an expert physician answering medical board exam questions accurately."

# 1. Base Qwen2.5-7B Closed-Book (batch_size=8)
print("[1/2] Evaluating Base Qwen2.5-7B Closed-Book...", flush=True)
base_preds = []
with torch.inference_mode(), model.disable_adapter():
    for start in range(0, N_TOTAL, 8):
        batch = []
        for k in range(start, min(start + 8, N_TOTAL)):
            ex = mcqa_val[k]
            um = (f"Question: {ex['question']}\nA) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}\n"
                  f"Answer with only the single letter (A, B, C, or D):")
            batch.append(tok.apply_chat_template([{"role": "system", "content": sys_msg}, {"role": "user", "content": um}], tokenize=False, add_generation_prompt=True))
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        lg = model(**enc, use_cache=False).logits[:, -1, :][:, letter_ids]
        base_preds.extend(torch.argmax(lg.float(), dim=-1).cpu().tolist())

# 2. MedVision-7B-v3-supermodel Single-Pass RAG & Cyclic-4 RAG
print("[2/2] Evaluating Single-Pass RAG & Cyclic-4 RAG...", flush=True)
rows_csv = []
base_ok = single_ok = cyclic_ok = 0
q_ex_cnt = q_8g_cnt = gopt_cnt = 0
dedup_n = dedup_single_ok = dedup_cyc_ok = 0
t1 = time.perf_counter()

with torch.inference_mode():
    for i, ex in enumerate(mcqa_val):
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        gold = int(ex["cop"])
        ctx = rag_ctxs[i]
        bp = base_preds[i]

        prompts = []
        for shift in range(4):
            rotated = [opts[(k + shift) % 4] for k in range(4)]
            user_msg = (
                f"Medical Reference Context:\n{ctx}\n\n"
                f"Question: {ex['question']}\n"
                f"A) {rotated[0]}\nB) {rotated[1]}\nC) {rotated[2]}\nD) {rotated[3]}\n"
                f"Answer with only the single letter (A, B, C, or D):"
            )
            prompts.append(tok.apply_chat_template([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True))

        probs_list = []
        for s in (0, 2):
            enc = tok(prompts[s:s+2], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            logits = model(**enc, use_cache=False).logits[:, -1, :][:, letter_ids]
            probs_list.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        all_p = np.vstack(probs_list)

        sp = int(np.argmax(all_p[0]))
        agg = np.zeros(4, dtype=np.float64)
        for shift in range(4):
            for k in range(4):
                agg[(k + shift) % 4] += all_p[shift, k]
        cp = int(np.argmax(agg))

        if bp == gold: base_ok += 1
        if sp == gold: single_ok += 1
        if cp == gold: cyclic_ok += 1

        nq, nc = norm_txt(ex["question"]), norm_txt(ctx)
        ngold = norm_txt(opts[gold])
        q_ex = int(len(nq) >= 15 and nq in nc)
        qw = nq.split()
        q_8g = int(len(qw) >= 8 and (" ".join(qw[:8]) in nc))
        gopt = int(len(ngold) >= 6 and ngold in nc)
        q_ex_cnt += q_ex; q_8g_cnt += q_8g; gopt_cnt += gopt
        if not (q_ex or q_8g):
            dedup_n += 1
            if sp == gold: dedup_single_ok += 1
            if cp == gold: dedup_cyc_ok += 1

        rows_csv.append({
            "idx": i,
            "question_id": str(ex.get("id", i)),
            "gold": letters[gold],
            "base_pred": letters[bp],
            "single_rag_pred": letters[sp],
            "cyclic4_pred": letters[cp],
            "q_exact_in_ctx": q_ex,
            "q_8gram_in_ctx": q_8g,
            "gold_opt_in_ctx": gopt,
        })
        if (i + 1) % 500 == 0 or (i + 1) == N_TOTAL:
            dt = time.perf_counter() - t1
            print(f"  [{i+1:4d}/{N_TOTAL}] Base:{100*base_ok/(i+1):.2f}% | Single:{100*single_ok/(i+1):.2f}% | Cyclic4:{100*cyclic_ok/(i+1):.2f}% ({dt/60:.1f}m)", flush=True)

csv_path = "./checkpoints/medmcqa_predictions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    w.writeheader()
    w.writerows(rows_csv)

summary = {
    "n_validation": N_TOTAL,
    "base_closed_book_pct": round(100.0 * base_ok / N_TOTAL, 2),
    "single_rag_pct": round(100.0 * single_ok / N_TOTAL, 2),
    "cyclic4_rag_pct": round(100.0 * cyclic_ok / N_TOTAL, 2),
    "q_exact_in_retrieved_ctx": q_ex_cnt,
    "q_8gram_in_retrieved_ctx": q_8g_cnt,
    "gold_opt_verbatim_in_retrieved_ctx": gopt_cnt,
    "dedup_n": dedup_n,
    "dedup_single_rag_pct": round(100.0 * dedup_single_ok / max(dedup_n, 1), 2),
    "dedup_cyclic4_rag_pct": round(100.0 * dedup_cyc_ok / max(dedup_n, 1), 2),
}
with open("./checkpoints/medmcqa_audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("SAVED checkpoints/medmcqa_predictions.csv | SUMMARY:", json.dumps(summary, indent=2))
