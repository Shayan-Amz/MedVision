import gc, io, json, os, re, time
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from app.services.vision_ensemble import analyze_chest_xray_ensemble
from app.services.clinical_llm_rag import synthesize_radiology_impression

def clean_deid(s):
    return re.sub(r"\s+", " ", (s or "").replace("XXXX", "")).strip()

raw_ds = load_dataset("ykumards/open-i", split="train").shuffle(seed=42)
print("Open-I columns:", raw_ds.column_names)

clean_recs = []
seen_uids = set()
for ex in raw_ds:
    uid = str(ex.get("uid", ""))
    if uid in seen_uids:
        continue
    fnd = clean_deid(ex.get("findings") or "")
    imp = clean_deid(ex.get("impression") or "")
    if len(fnd) >= 20 and len(imp) >= 8 and fnd.lower() != "none" and imp.lower() != "none":
        seen_uids.add(uid)
        clean_recs.append(ex)

test_recs = clean_recs[:250]
train_recs = clean_recs[250:]
test_uids = {str(r.get("uid", "")) for r in test_recs}
train_uids = {str(r.get("uid", "")) for r in train_recs}
assert len(test_uids & train_uids) == 0, "IU-Xray UID overlap detected!"
print(f"IU-Xray Zero-Overlap Verified: Test={len(test_recs)} UIDs, Train={len(train_recs)} UIDs, Overlap=0")

NEG_HINTS = ("no acute", "normal", "unremarkable", "clear lungs", "no active", "no cardiopulmonary")
POS_HINTS = ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")

def to_pil(raw_img):
    if isinstance(raw_img, Image.Image):
        return raw_img.convert("RGB")
    if isinstance(raw_img, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw_img)).convert("RGB")
    if isinstance(raw_img, dict) and "bytes" in raw_img:
        return Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
    return None

items = []
y_heur, y_mesh = [], []
scores_maxp, preds_tri, preds_bal, preds_alert = [], [], [], []
latencies_ms = []
lexical_contradictions = 0
ood_false_rejects = 0
ctr_gt50_total = 0
ctr_gt50_mentioned = 0
skipped_no_img = 0

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

for idx, ex in enumerate(test_recs):
    uid = str(ex.get("uid", ""))
    pil_img = to_pil(ex.get("img_frontal"))
    if pil_img is None:
        skipped_no_img += 1
        continue

    t0 = time.perf_counter()
    v_tri = analyze_chest_xray_ensemble(pil_img, operating_mode="triage")
    v_bal = analyze_chest_xray_ensemble(pil_img, operating_mode="balanced")
    if v_bal.get("ood_rejected"):
        ood_false_rejects += 1

    top_findings = [(f["name"], f["prob"]) for f in v_bal["findings"] if f["prob"] >= 0.45] or v_bal["top3"]
    ctr_str = str(v_bal["ctr"]) if v_bal["ctr"] is not None else "N/A"
    lres = synthesize_radiology_impression(
        primary=v_bal["primary_finding"],
        top_conf=v_bal["top_confidence"],
        top3=top_findings,
        urgency=v_bal["urgency"],
        patient_ctx=f"IU-Xray Held-Out UID={uid}, Exact PSPNet CTR={ctr_str}",
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(dt_ms)

    rep_txt = lres.get("report") or ""
    rep_low = rep_txt.lower()
    pos_vis = [n for n, p in top_findings if p >= 0.60]
    is_contra = any(f"no {n.lower().replace('_', ' ')}" in rep_low for n in pos_vis)
    if is_contra:
        lexical_contradictions += 1

    if v_bal["ctr"] is not None and float(v_bal["ctr"]) > 0.50:
        ctr_gt50_total += 1
        if any(k in rep_low for k in ("cardiomegaly", "enlarged", "ctr")):
            ctr_gt50_mentioned += 1

    gt_imp = clean_deid(ex.get("impression") or "").lower()
    is_gt_norm = any(k in gt_imp for k in NEG_HINTS) and not any(k in gt_imp for k in POS_HINTS)
    yh = 0 if is_gt_norm else 1
    y_heur.append(yh)

    mesh_raw = str(ex.get("Problems") or ex.get("problems") or ex.get("MeSH") or "").strip().lower()
    ym = (0 if mesh_raw in ("normal", "normal;", "none") else 1) if mesh_raw else None
    if ym is not None:
        y_mesh.append(ym)

    max_p = max((f["prob"] for f in v_bal["findings"]), default=0.0)
    scores_maxp.append(float(max_p))

    tri_pos = 1 if any(f.get("positive") for f in v_tri["findings"]) else 0
    bal_pos = 1 if any(f.get("positive") for f in v_bal["findings"]) else 0
    alert_pos = 1 if (bal_pos == 1 and float(v_bal["top_confidence"]) >= 0.62) else 0
    preds_tri.append(tri_pos); preds_bal.append(bal_pos); preds_alert.append(alert_pos)

    items.append({
        "idx": idx, "uid": uid, "in_radiologist_train": uid in train_uids,
        "gt_impression": clean_deid(ex.get("impression") or ""),
        "gt_report_heuristic_abnormal": yh,
        "gt_problems_col_abnormal": ym,
        "max_prob": round(float(max_p), 4),
        "primary_finding": v_bal["primary_finding"],
        "top_confidence": round(float(v_bal["top_confidence"]), 4),
        "ctr": v_bal["ctr"],
        "ood_rejected": bool(v_bal.get("ood_rejected", False)),
        "triage_pos": tri_pos, "balanced_pos": bal_pos, "alert_gate_pos": alert_pos,
        "lexical_contradiction": is_contra,
        "llm_report": rep_txt,
        "latency_ms": round(dt_ms, 1),
    })

peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

def bin_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sens_pct": round(100.0 * tp / max(tp + fn, 1), 1),
            "spec_pct": round(100.0 * tn / max(tn + fp, 1), 1),
            "ppv_pct": round(100.0 * tp / max(tp + fp, 1), 1)}

auroc_h = round(float(roc_auc_score(y_heur, scores_maxp)), 4)
rng = np.random.default_rng(42)
boot_aucs = []
ya, sa = np.array(y_heur), np.array(scores_maxp)
for _ in range(2000):
    b = rng.integers(0, len(ya), len(ya))
    if len(np.unique(ya[b])) == 2:
        boot_aucs.append(float(roc_auc_score(ya[b], sa[b])))
auroc_ci = [round(float(np.percentile(boot_aucs, 2.5)), 4), round(float(np.percentile(boot_aucs, 97.5)), 4)]

summary = {
    "dataset": "ykumards/open-i (Indiana University IU-Xray)",
    "split_protocol": "Exact phase3b held-out test split: shuffle(seed=42), unique UIDs with len(findings)>=20 & len(impression)>=8, test=[:250], train=[250:]",
    "uid_overlap_with_radiologist_train": 0,
    "n_test_uids": len(test_recs),
    "skipped_missing_frontal_img": skipped_no_img,
    "n_evaluated": len(items),
    "n_abnormal_heuristic": int(sum(y_heur)),
    "n_normal_heuristic": int(len(y_heur) - sum(y_heur)),
    "label_method": "Report-impression keyword heuristic (NOT MeSH or blinded multi-reader ground truth)",
    "contradiction_method": "Lexical negation pattern check ('no <finding>') on high-confidence (>=0.60) vision findings (NOT full clinical NLI)",
    "auroc_max_p": auroc_h,
    "auroc_ci95_boot2000": auroc_ci,
    "er_triage": bin_metrics(y_heur, preds_tri),
    "balanced_mode": bin_metrics(y_heur, preds_bal),
    "high_spec_alert_gate": bin_metrics(y_heur, preds_alert),
    "lexical_contradictions": lexical_contradictions,
    "ood_frontal_false_rejects": ood_false_rejects,
    "ctr_gt50_total": ctr_gt50_total,
    "ctr_gt50_mentioned": ctr_gt50_mentioned,
    "mean_latency_ms": round(float(np.mean(latencies_ms)), 1),
    "peak_vram_gb": round(float(peak_vram_gb), 2),
    "items": items,
}
with open("./checkpoints/iu_xray_e2e_audit.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("IU-XRAY ZERO-OVERLAP E2E SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))
