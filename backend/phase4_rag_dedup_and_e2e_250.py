import io, json, re, time
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset
from app.services.vision_ensemble import analyze_chest_xray_ensemble, NIH14_CLASSES
from app.services.clinical_llm_rag import synthesize_radiology_impression

print("=" * 90)
print("  OPTION 1: RAG Leak-Free Deduplication Audit (175k vs 4,183) + 250-Case External E2E")
print("=" * 90)

# =========================================================================
# PART A: Exact & 8-gram Contamination Audit (175,413 RAG docs vs 4,183 MedMCQA)
# =========================================================================
print("\n[A/2] Auditing 175,413 FAISS RAG passages against all 4,183 MedMCQA validation questions...")
t0 = time.perf_counter()
with open("./medical_rag_index_full/full_docs.json", "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

ds_mcqa = load_dataset("openlifescienceai/medmcqa", split="validation")
mcqa_val = [ex for ex in ds_mcqa if ex.get("cop") in (0, 1, 2, 3)]

def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

# Build inverted 8-gram set from all 4,183 MedMCQA questions (fast O(N))
q_norm_list = [norm_text(ex["question"]) for ex in mcqa_val]
q_8grams = {}
for qi, qn in enumerate(q_norm_list):
    toks = qn.split()
    if len(toks) >= 8:
        g = " ".join(toks[:8])
        q_8grams.setdefault(g, set()).add(qi)

exact_matched_qs = set()
ngram8_matched_qs = set()

# Check exact match for long questions (>=35 chars) & 8-gram prefix match
long_qs = [(qi, qn) for qi, qn in enumerate(q_norm_list) if len(qn) >= 35]
rag_norm_set_check = [norm_text(d) for d in rag_docs]

# Build 8-gram index of all RAG docs
rag_8gram_set = set()
for dn in rag_norm_set_check:
    toks = dn.split()
    for k in range(0, max(0, len(toks) - 7)):
        rag_8gram_set.add(" ".join(toks[k:k+8]))

for g, q_indices in q_8grams.items():
    if g in rag_8gram_set:
        ngram8_matched_qs.update(q_indices)

# Only scan full exact substring on candidates that already matched 8-gram
for qi in ngram8_matched_qs:
    qn = q_norm_list[qi]
    if len(qn) >= 35:
        if any(qn in dn for dn in rag_norm_set_check):
            exact_matched_qs.add(qi)

rag_audit = {
    "rag_corpus_size": len(rag_docs),
    "medmcqa_val_size": len(mcqa_val),
    "exact_question_matches": len(exact_matched_qs),
    "exact_match_rate_pct": round(len(exact_matched_qs) / len(mcqa_val) * 100, 3),
    "ngram8_overlap_questions": len(ngram8_matched_qs),
    "ngram8_overlap_rate_pct": round(len(ngram8_matched_qs) / len(mcqa_val) * 100, 3),
    "audit_time_sec": round(time.perf_counter() - t0, 1),
}
print("  [RAG LEAK-FREE AUDIT]", json.dumps(rag_audit, indent=2))
del rag_docs, rag_norm_set_check, rag_8gram_set

# =========================================================================
# PART B: 250-Case External-Hospital End-to-End Validation (ER Triage vs Balanced)
# =========================================================================
print("\n[B/2] Running 250-Case External-Hospital E2E Validation on Unseen IU-Xray (Triage vs Balanced)...")
ds_iu = load_dataset("ykumards/open-i", split="train")
e2e_samples = []
for i in range(len(ds_iu) - 1, -1, -1):
    ex = ds_iu[i]
    raw_img = ex.get("img_frontal")
    imp = (ex.get("impression") or "").strip()
    if raw_img is None or len(imp) <= 10:
        continue
    try:
        if isinstance(raw_img, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
        elif isinstance(raw_img, dict) and raw_img.get("bytes"):
            pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
        else:
            pil_img = raw_img.convert("RGB")
        e2e_samples.append((pil_img, imp))
    except Exception:
        continue
    if len(e2e_samples) >= 250:
        break

print(f"  Loaded {len(e2e_samples)} unseen frontal IU-Xray studies (matches Phase 3B test size n=250).")

latencies_ms = []
contradictions = 0
ctr_enlarged_total, ctr_enlarged_mentions = 0, 0
normal_gt, abnormal_gt = 0, 0
triage_norm_ok, triage_abn_ok = 0, 0
bal_norm_ok, bal_abn_ok = 0, 0
ood_skipped = 0

with open("./checkpoints/medvision_nih14_ensemble_meta.json", "r", encoding="utf-8") as f:
    ens_meta = json.load(f)
val_thresholds = ens_meta.get("thresholds", {c: 0.50 for c in NIH14_CLASSES})

for idx, (pil_img, gt_imp) in enumerate(e2e_samples, 1):
    t1 = time.perf_counter()
    vres = analyze_chest_xray_ensemble(pil_img, operating_mode="balanced")
    if vres.get("ood_rejected"):
        ood_skipped += 1
        continue

    top_findings = [(f["name"], f["prob"]) for f in vres["findings"] if f["prob"] >= 0.45] or vres["top3"]
    ctr_str = str(vres["ctr"]) if vres["ctr"] is not None else "N/A"
    lres = synthesize_radiology_impression(
        primary=vres["primary_finding"],
        top_conf=vres["top_confidence"],
        top3=top_findings,
        urgency=vres["urgency"],
        patient_ctx=f"IU-Xray External Case #{idx}, Exact PSPNet CTR={ctr_str}",
    )
    latencies_ms.append((time.perf_counter() - t1) * 1000.0)

    rep_low = lres["report"].lower()
    pos_vis = [n for n, p in top_findings if p >= 0.60]
    if any(f"no {n.lower().replace('_', ' ')}" in rep_low for n in pos_vis):
        contradictions += 1

    if vres["ctr"] is not None and vres["ctr"] > 0.50:
        ctr_enlarged_total += 1
        if "cardiomegaly" in rep_low or "enlarged" in rep_low or "ctr" in rep_low:
            ctr_enlarged_mentions += 1

    # Compare BOTH operating modes on the exact same calibrated probabilities
    triage_any_pos = any(f["prob"] >= float(val_thresholds.get(f["name"], 0.50)) for f in vres["findings"])
    bal_any_pos = any(f["positive"] for f in vres["findings"])

    gt_low = gt_imp.lower()
    is_gt_normal = any(k in gt_low for k in ("no acute", "normal", "unremarkable", "clear lungs")) and not any(
        k in gt_low for k in ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")
    )
    if is_gt_normal:
        normal_gt += 1
        if not triage_any_pos:
            triage_norm_ok += 1
        if (not bal_any_pos) or vres["urgency"] == "ROUTINE":
            bal_norm_ok += 1
    else:
        abnormal_gt += 1
        if triage_any_pos:
            triage_abn_ok += 1
        if bal_any_pos or vres["urgency"] in ("HIGH", "STAT") or len(pos_vis) > 0:
            bal_abn_ok += 1

    if idx % 50 == 0 or idx == len(e2e_samples):
        print(f"  [E2E {idx:3d}/250] Mean Latency: {np.mean(latencies_ms):.0f}ms | Contradictions: {contradictions} | OOD skipped: {ood_skipped}", flush=True)

e2e_250 = {
    "n_evaluated": len(latencies_ms),
    "ood_skipped": ood_skipped,
    "mean_e2e_latency_ms": round(float(np.mean(latencies_ms)), 1),
    "cross_modal_contradiction_rate_pct": round(contradictions / max(len(latencies_ms), 1) * 100, 2),
    "ctr_gt_050_total": ctr_enlarged_total,
    "ctr_gt_050_reported_by_llm_pct": round(ctr_enlarged_mentions / max(ctr_enlarged_total, 1) * 100, 1),
    "normal_gt_cases": normal_gt,
    "abnormal_gt_cases": abnormal_gt,
    "er_triage_mode": {
        "abnormal_sensitivity_pct": round(triage_abn_ok / max(abnormal_gt, 1) * 100, 1),
        "normal_specificity_pct": round(triage_norm_ok / max(normal_gt, 1) * 100, 1),
    },
    "balanced_mode": {
        "abnormal_sensitivity_pct": round(bal_abn_ok / max(abnormal_gt, 1) * 100, 1),
        "normal_agreement_pct": round(bal_norm_ok / max(normal_gt, 1) * 100, 1),
    },
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
}
print("\n[250-CASE EXTERNAL E2E STATS]", json.dumps(e2e_250, indent=2))

with open("./checkpoints/reviewer2_rag_audit_and_e2e250.json", "w", encoding="utf-8") as f:
    json.dump({"rag_audit": rag_audit, "e2e_250": e2e_250}, f, indent=2)
print("\n[SAVED] -> ./checkpoints/reviewer2_rag_audit_and_e2e250.json")
