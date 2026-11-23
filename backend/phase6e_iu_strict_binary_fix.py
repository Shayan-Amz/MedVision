import io, json
import numpy as np
from PIL import Image
from datasets import load_dataset
from app.services.vision_ensemble import analyze_chest_xray_ensemble, NIH14_CLASSES

with open("./checkpoints/medvision_nih14_ensemble_meta.json", "r", encoding="utf-8") as f:
    ens_meta = json.load(f)
val_thr = ens_meta.get("thresholds", {c: 0.50 for c in NIH14_CLASSES})

ds_iu = load_dataset("ykumards/open-i", split="train")
rows = []
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
    vres = analyze_chest_xray_ensemble(pil_img, operating_mode="balanced")
    if vres.get("ood_rejected"):
        continue
    gt_low = imp.lower()
    is_norm = any(k in gt_low for k in ("no acute", "normal", "unremarkable", "clear lungs")) and not any(
        k in gt_low for k in ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")
    )
    gt_abn = 0 if is_norm else 1
    p_triage = int(any(f["prob"] >= float(val_thr.get(f["name"], 0.50)) for f in vres["findings"]))
    p_bal    = int(any(f["positive"] for f in vres["findings"]))  # strict p_c >= t_c + 0.15
    p_urg    = int(vres["urgency"] in ("HIGH", "STAT"))           # strict HIGH/STAT vs ROUTINE
    rows.append((gt_abn, p_triage, p_bal, p_urg))
    if len(rows) >= 250:
        break

arr = np.array(rows)
y = arr[:, 0]

def bin_metrics(pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = round(tp / max(tp + fn, 1) * 100, 1)
    spec = round(tn / max(tn + fp, 1) * 100, 1)
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "sensitivity_pct": sens, "specificity_pct": spec}

res = {
    "n_total": int(len(y)),
    "n_abnormal_gt": int((y == 1).sum()),
    "n_normal_gt": int((y == 0).sum()),
    "er_triage_strict_binary": bin_metrics(arr[:, 1]),
    "balanced_shift015_strict_binary": bin_metrics(arr[:, 2]),
    "urgency_gate_high_stat_vs_routine": bin_metrics(arr[:, 3]),
}
print(json.dumps(res, indent=2))
