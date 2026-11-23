import csv, gc, json, os, re, time, faiss
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. PubMedQA Canonical Split Manifest (800 Train / 200 Test pubids)
ds_pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
test_pubids = [str(r["pubid"]) for r in ds_pqa.select(range(0, 200))]
train_pubids = [str(r["pubid"]) for r in ds_pqa.select(range(200, 1000))]
assert len(set(test_pubids) & set(train_pubids)) == 0
with open("./checkpoints/pubmedqa_split_manifest.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "qiaojin/PubMedQA (pqa_labeled)",
        "seed": 42,
        "train_range": [200, 1000],
        "test_range": [0, 200],
        "overlap_count": 0,
        "train_pubids": train_pubids,
        "test_pubids": test_pubids,
    }, f, indent=2)
print(f"[1/2] Saved checkpoints/pubmedqa_split_manifest.json (Train={len(train_pubids)}, Test={len(test_pubids)}, Overlap=0)", flush=True)

# 2. MedMCQA 4-Option-in-Context Audit & Retrieved Passage Provenance
def norm_txt(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()

ds_mcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
mcqa_val = [ex for ex in ds_mcqa if ex.get("cop") in (0, 1, 2, 3)]
with open("./checkpoints/medmcqa_predictions.csv", "r", encoding="utf-8") as f:
    pred_rows = list(csv.DictReader(f))
assert len(pred_rows) == len(mcqa_val)

rag_dir = "./medical_rag_index_full"
index = faiss.read_index(os.path.join(rag_dir, "full_faiss.index"))
with open(os.path.join(rag_dir, "full_docs.json"), "r", encoding="utf-8") as f:
    docs = json.load(f)

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
del reranker; gc.collect(); torch.cuda.empty_cache()

# Analyze all 4 options (A/B/C/D) presence in retrieved Top-3 context & inspect corpus provenance
gold_in = 0
dist_in_any = 0
dist_slot_hits = 0
dist_slot_total = 0
qa_pattern_hits = 0
samples_gold_in_ctx = []
letters = ["A", "B", "C", "D"]

# Check how many docs look like textbook vs Q&A format
sample_doc_prefixes = [docs[k][:120] for k in range(0, len(docs), max(1, len(docs)//15))][:15]

for i, ex in enumerate(mcqa_val):
    cand_idxs = I[i]
    cand_scores = scores[i * 20 : (i + 1) * 20]
    top3_ids = [int(cand_idxs[j]) for j in np.argsort(cand_scores)[::-1][:3]]
    top3_txts = [docs[d_id][:450] for d_id in top3_ids]
    ctx_raw = "\n".join(top3_txts)
    nc = norm_txt(ctx_raw)

    opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
    gold = int(ex["cop"])
    opt_hits = [int(len(norm_txt(o)) >= 6 and norm_txt(o) in nc) for o in opts]
    g_hit = opt_hits[gold]
    d_hits = [opt_hits[k] for k in range(4) if k != gold]
    gold_in += g_hit
    if any(d_hits): dist_in_any += 1
    dist_slot_hits += sum(d_hits)
    dist_slot_total += 3

    if re.search(r"\b(correct answer|answer\s*[:\-]\s*[a-d]|option\s+[a-d]\s+is\s+correct|ans\s*[:\.])\b", ctx_raw.lower()):
        qa_pattern_hits += 1

    if g_hit == 1 and len(samples_gold_in_ctx) < 25:
        samples_gold_in_ctx.append({
            "idx": i,
            "question_id": str(ex.get("id", i)),
            "question": ex["question"],
            "gold_letter": letters[gold],
            "gold_option": opts[gold],
            "all_options_in_ctx": {letters[k]: opt_hits[k] for k in range(4)},
            "cyclic4_correct": pred_rows[i]["cyclic4_pred"] == letters[gold],
            "top3_doc_ids": top3_ids,
            "top3_passages": top3_txts,
        })

audit = {
    "n_validation": len(mcqa_val),
    "corpus_total_passages": len(docs),
    "corpus_sample_prefixes": sample_doc_prefixes,
    "gold_option_in_ctx_count": gold_in,
    "gold_option_in_ctx_pct": round(100.0 * gold_in / len(mcqa_val), 2),
    "any_distractor_in_ctx_count": dist_in_any,
    "any_distractor_in_ctx_pct": round(100.0 * dist_in_any / len(mcqa_val), 2),
    "per_distractor_slot_in_ctx_pct": round(100.0 * dist_slot_hits / dist_slot_total, 2),
    "retrieved_ctx_with_explicit_qa_answer_key_pattern": qa_pattern_hits,
    "samples_where_gold_in_ctx": samples_gold_in_ctx,
}
with open("./checkpoints/medmcqa_rag_provenance_and_options_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("[2/2] MEDMCQA 4-OPTION & PROVENANCE AUDIT:", json.dumps({k: v for k, v in audit.items() if k not in ("samples_gold_in_ctx", "corpus_sample_prefixes")}, indent=2))
print("Sample Corpus Doc Prefixes (first 5):")
for p in sample_doc_prefixes[:5]:
    print("  -", p.replace("\n", " "))
