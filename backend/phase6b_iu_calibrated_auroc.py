import io, json
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from app.services.vision_ensemble import analyze_chest_xray_ensemble, NIH14_CLASSES

with open("./checkpoints/medvision_nih14_ensemble_meta.json", "r", encoding="utf-8") as f:
    ens_meta = json.load(f)
val_thr = ens_meta.get("thresholds", {c: 0.50 for c in NIH14_CLASSES})

ds_iu = load_dataset("ykumards/open-i", split="train")
iu_y, raw_scores, cal_scores = [], [], []
cnt = 0
for i in range(len(ds_iu) - 1, -1, -1):
    ex = ds_iu[i]
    raw_img = ex.get("img_frontal")
    imp = (ex.get("impression") or "").strip()
    if raw_img is None or len(imp) <= 10:
        continue
    try:
        pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB") if isinstance(raw_img, (bytes, bytearray)) else raw_img.convert("RGB")
    except Exception:
        continue
    vres = analyze_chest_xray_ensemble(pil_img, operating_mode="triage")
    if vres.get("ood_rejected"):
        continue
    gt_low = imp.lower()
    is_norm = any(k in gt_low for k in ("no acute", "normal", "unremarkable", "clear lungs")) and not any(
        k in gt_low for k in ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")
    )
    iu_y.append(0 if is_norm else 1)
    raw_scores.append(float(vres["top_confidence"]))
    margin = max(float(f["prob"]) - float(val_thr.get(f["name"], 0.50)) for f in vres["findings"])
    cal_scores.append(margin)
    cnt += 1
    if cnt >= 250:
        break

iu_y = np.array(iu_y)
raw_scores = np.array(raw_scores)
cal_scores = np.array(cal_scores)

rng = np.random.default_rng(42)
auc_raw = float(roc_auc_score(iu_y, raw_scores))
auc_cal = float(roc_auc_score(iu_y, cal_scores))
b_cal = [roc_auc_score(iu_y[idx], cal_scores[idx]) for idx in (rng.integers(0, len(iu_y), len(iu_y)) for _ in range(2000)) if len(np.unique(iu_y[idx])) == 2]

res = {
    "n_studies": int(len(iu_y)),
    "raw_max_p_auroc": round(auc_raw, 4),
    "calibrated_margin_auroc": round(auc_cal, 4),
    "calibrated_ci95_B2000": [round(float(np.percentile(b_cal, 2.5)), 4), round(float(np.percentile(b_cal, 97.5)), 4)],
}
print(json.dumps(res, indent=2))
