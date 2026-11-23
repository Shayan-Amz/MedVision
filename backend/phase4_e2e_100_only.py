import io, json, time
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset
from app.services.vision_ensemble import analyze_chest_xray_ensemble
from app.services.clinical_llm_rag import synthesize_radiology_impression

print("=" * 88)
print("  [PART 3 ONLY] 100-Case End-to-End Multimodal Validation (Raw IU-Xray -> Vision+CTR -> LLM)")
print("=" * 88)

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
    if len(e2e_samples) >= 100:
        break

print(f"Loaded {len(e2e_samples)} unseen frontal IU-Xray studies.")
latencies_ms = []
contradictions = 0
normal_gt, normal_pred_ok = 0, 0
abnormal_gt, abnormal_detected = 0, 0
ctr_enlarged_mentions, ctr_enlarged_total = 0, 0

for idx, (pil_img, gt_imp) in enumerate(e2e_samples, 1):
    t0 = time.perf_counter()
    vres = analyze_chest_xray_ensemble(pil_img, operating_mode="balanced")
    if vres.get("ood_rejected"):
        continue
    top_findings = [(f["name"], f["prob"]) for f in vres["findings"] if f["prob"] >= 0.45] or vres["top3"]
    ctr_str = str(vres["ctr"]) if vres["ctr"] is not None else "N/A"
    lres = synthesize_radiology_impression(
        primary=vres["primary_finding"],
        top_conf=vres["top_confidence"],
        top3=top_findings,
        urgency=vres["urgency"],
        patient_ctx=f"IU-Xray Test Case #{idx}, Exact PSPNet CTR={ctr_str}",
    )
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    rep_low = lres["report"].lower()
    pos_vis = [n for n, p in top_findings if p >= 0.60]
    if any(f"no {n.lower().replace('_', ' ')}" in rep_low for n in pos_vis):
        contradictions += 1

    if vres["ctr"] is not None and vres["ctr"] > 0.50:
        ctr_enlarged_total += 1
        if "cardiomegaly" in rep_low or "enlarged" in rep_low or "ctr" in rep_low:
            ctr_enlarged_mentions += 1

    gt_low = gt_imp.lower()
    is_gt_normal = any(k in gt_low for k in ("no acute", "normal", "unremarkable", "clear lungs")) and not any(
        k in gt_low for k in ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")
    )
    if is_gt_normal:
        normal_gt += 1
        if vres["urgency"] == "ROUTINE" or "no acute" in rep_low or "normal" in rep_low:
            normal_pred_ok += 1
    else:
        abnormal_gt += 1
        if vres["urgency"] in ("HIGH", "STAT") or len(pos_vis) > 0:
            abnormal_detected += 1

    if idx % 25 == 0:
        print(f"  [E2E {idx}/100] Mean Latency: {np.mean(latencies_ms):.0f}ms | Contradictions: {contradictions}", flush=True)

e2e_stats = {
    "n_evaluated": len(latencies_ms),
    "mean_e2e_latency_ms": round(float(np.mean(latencies_ms)), 1),
    "cross_modal_contradiction_rate_pct": round(contradictions / max(len(latencies_ms), 1) * 100, 2),
    "ctr_gt_050_total": ctr_enlarged_total,
    "ctr_gt_050_reported_by_llm_pct": round(ctr_enlarged_mentions / max(ctr_enlarged_total, 1) * 100, 1),
    "normal_gt_cases": normal_gt,
    "normal_triage_agreement_pct": round(normal_pred_ok / max(normal_gt, 1) * 100, 1),
    "abnormal_gt_cases": abnormal_gt,
    "abnormal_triage_sensitivity_pct": round(abnormal_detected / max(abnormal_gt, 1) * 100, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
}
print("\n[E2E MULTIMODAL STATS]", json.dumps(e2e_stats, indent=2))
with open("./checkpoints/reviewer2_e2e_100.json", "w", encoding="utf-8") as f:
    json.dump(e2e_stats, f, indent=2)
