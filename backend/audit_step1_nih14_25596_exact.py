import gc, json, os
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from huggingface_hub import HfApi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("./nih_cxr14_cache/splits_meta.json", "r", encoding="utf-8") as f:
    nih_meta = json.load(f)
with open("./checkpoints/medvision_nih14_ensemble_meta.json", "r", encoding="utf-8") as f:
    ens_meta = json.load(f)

classes = nih_meta.get("classes", [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia",
    "Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
])
cls_to_idx = {c: i for i, c in enumerate(classes)}
img_dir = "./nih_cxr14_cache/images"

# 1. Load 96 tail test rows with EXACT label_names & unified 8-digit patient_id
api = HfApi()
all_files = api.list_repo_files("timm/nih-chest-xray-14", repo_type="dataset")
test_shards = sorted([f for f in all_files if "test-" in f and f.endswith(".parquet")])
ds_last = load_dataset("timm/nih-chest-xray-14", data_files={"test": test_shards[-1]}, split="test")
hf_label_names = ds_last.features["labels"].feature.names
extra_rows = ds_last.select(range(len(ds_last) - 96, len(ds_last)))

extra_imgs, extra_labels, extra_pids, extra_iids = [], [], [], []
for row in extra_rows:
    extra_imgs.append(row["image"].convert("RGB"))
    extra_pids.append(str(int(row["patient_id"])).zfill(8))
    extra_iids.append(str(row["image_id"]))
    vec = np.zeros(len(classes), dtype=np.float32)
    names = row.get("label_names")
    if names is None:
        names = [hf_label_names[int(idx)] for idx in (row.get("labels") or [])]
    for nm in names:
        if nm in cls_to_idx:
            vec[cls_to_idx[nm]] = 1.0
    extra_labels.append(vec)

cached_iids = [str(r["file"]).rsplit(".", 1)[0] for r in nih_meta["test"]]
cached_pids = [str(int(str(r["file"]).split("_")[0])).zfill(8) for r in nih_meta["test"]]
image_ids = cached_iids + extra_iids
patient_ids = cached_pids + extra_pids

# Verify ZERO duplicate images and ZERO patient leakage across Train/Val/Test
assert len(image_ids) == 25596, f"Expected 25596, got {len(image_ids)}"
assert len(set(image_ids)) == 25596, f"Duplicate image_ids found: {25596 - len(set(image_ids))}"
train_pids = {str(int(str(r["file"]).split("_")[0])).zfill(8) for r in nih_meta["train"]}
val_pids = {str(int(str(r["file"]).split("_")[0])).zfill(8) for r in nih_meta["val"]}
test_pids_set = set(patient_ids)
assert len(train_pids & test_pids_set) == 0, "Train/Test patient overlap!"
assert len(val_pids & test_pids_set) == 0, "Val/Test patient overlap!"
assert len(train_pids & val_pids) == 0, "Train/Val patient overlap!"
print(f"Leakage Check PASSED: 0 overlap | Train={len(train_pids):,} pts, Val={len(val_pids):,} pts, Test={len(test_pids_set):,} pts ({len(image_ids):,} imgs)")

class NIHDS(Dataset):
    def __init__(self, recs, tf): self.recs = recs; self.tf = tf
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return self.tf(Image.open(os.path.join(img_dir, r["file"])).convert("RGB")), torch.tensor(r["labels"], dtype=torch.float32)

tf224 = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf256 = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@torch.no_grad()
def infer_all(model, recs, extra_pils, tf):
    model.eval(); ps, ys = [], []
    for x, y in DataLoader(NIHDS(recs, tf), batch_size=64, shuffle=False, num_workers=0):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = 0.5 * (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    bx = torch.stack([tf(im) for im in extra_pils]).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ep = 0.5 * (torch.sigmoid(model(bx).float()) + torch.sigmoid(model(torch.flip(bx, dims=[3])).float()))
    ps.append(ep.cpu().numpy()); ys.append(np.vstack(extra_labels))
    return np.vstack(ps), np.vstack(ys)

dn_ckpt = torch.load("./checkpoints/medvision_nih14_densenet121.pth", map_location=device, weights_only=False)
dn_state = dn_ckpt["model_state_dict"] if "model_state_dict" in dn_ckpt else dn_ckpt
dn = tv_models.densenet121(weights=None)
in_dn = dn.classifier.in_features
dn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dn, len(classes))) if "classifier.1.weight" in dn_state else nn.Linear(in_dn, len(classes))
dn.load_state_dict(dn_state); dn = dn.to(device).eval()
p_dn, y_all = infer_all(dn, nih_meta["test"], extra_imgs, tf224)
del dn; gc.collect(); torch.cuda.empty_cache()

cn_ckpt = torch.load("./checkpoints/medvision_nih14_convnext_small.pth", map_location=device, weights_only=False)
cn = tv_models.convnext_small(weights=None)
cn.classifier[2] = nn.Sequential(nn.Dropout(0.2), nn.Linear(cn.classifier[2].in_features, len(classes)))
cn.load_state_dict(cn_ckpt["model_state_dict"]); cn = cn.to(device).eval()
p_cn, _ = infer_all(cn, nih_meta["test"], extra_imgs, tf256)
del cn; gc.collect(); torch.cuda.empty_cache()

w_cn = float(ens_meta.get("convnext_weight", 0.45))
w_dn = float(ens_meta.get("densenet_weight", 0.55))
p_ens = w_cn * p_cn + w_dn * p_dn
thrs_raw = ens_meta.get("thresholds", {})
thr_vec = np.array([float(thrs_raw[c]) if isinstance(thrs_raw, dict) else float(thrs_raw[i]) for i, c in enumerate(classes)], dtype=np.float32)

# Save raw auditable predictions NPZ
np.savez_compressed(
    "./checkpoints/nih14_test_predictions.npz",
    y_true=y_all.astype(np.uint8),
    p_dn=p_dn.astype(np.float32),
    p_cn=p_cn.astype(np.float32),
    p_ens=p_ens.astype(np.float32),
    image_ids=np.array(image_ids),
    patient_ids=np.array(patient_ids),
    classes=np.array(classes),
    thresholds=thr_vec,
    weights=np.array([w_dn, w_cn], dtype=np.float32),
)

# Per-class exact metrics
per_class = []
for i, cls in enumerate(classes):
    yt = y_all[:, i]
    a_dn = float(roc_auc_score(yt, p_dn[:, i]))
    a_cn = float(roc_auc_score(yt, p_cn[:, i]))
    a_en = float(roc_auc_score(yt, p_ens[:, i]))
    yp = (p_ens[:, i] >= thr_vec[i]).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    sens = 100.0 * tp / max(tp + fn, 1); spec = 100.0 * tn / max(tn + fp, 1)
    per_class.append({"class": cls, "dn": round(a_dn, 4), "cn": round(a_cn, 4), "ens": round(a_en, 4),
                      "sens": round(sens, 1), "spec": round(spec, 1), "fpr": round(100.0-spec, 1), "fnr": round(100.0-sens, 1)})

m_dn = float(np.mean([r["dn"] for r in per_class]))
m_cn = float(np.mean([r["cn"] for r in per_class]))
m_en = float(np.mean([r["ens"] for r in per_class]))
m_sens = float(np.mean([r["sens"] for r in per_class]))
m_spec = float(np.mean([r["spec"] for r in per_class]))

with open("./checkpoints/nih14_full_25596_per_class.json", "w", encoding="utf-8") as f:
    json.dump({"n_official_test": 25596, "n_unique_patients": len(test_pids_set),
               "mean_dn": round(m_dn, 4), "mean_cn": round(m_cn, 4), "mean_ens": round(m_en, 4),
               "mean_sens": round(m_sens, 1), "mean_spec": round(m_spec, 1), "per_class": per_class}, f, indent=2)

# 2. Patient-Level Cluster Paired Bootstrap (B=2,000) with Paired Delta-AUROC 95% CI
pid_groups = defaultdict(list)
for i, pid in enumerate(patient_ids):
    pid_groups[pid].append(i)
group_arrays = [np.array(v, dtype=np.int32) for v in pid_groups.values()]
n_patients = len(group_arrays)

rng = np.random.default_rng(42)
s_dn, s_cn, s_en, d_dn, d_cn = [], [], [], [], []
for _ in range(2000):
    sampled_g = rng.integers(0, n_patients, size=n_patients)
    idx = np.concatenate([group_arrays[g] for g in sampled_g])
    yb = y_all[idx]
    if np.any(yb.sum(axis=0) == 0) or np.any(yb.sum(axis=0) == len(yb)):
        continue
    ad = float(roc_auc_score(yb, p_dn[idx], average="macro"))
    ac = float(roc_auc_score(yb, p_cn[idx], average="macro"))
    ae = float(roc_auc_score(yb, p_ens[idx], average="macro"))
    s_dn.append(ad); s_cn.append(ac); s_en.append(ae)
    d_dn.append(ae - ad); d_cn.append(ae - ac)

B_valid = len(s_en)
boot_res = {
    "n_images": 25596,
    "n_unique_patients": int(n_patients),
    "B_valid": int(B_valid),
    "point_dn": round(m_dn, 4), "point_cn": round(m_cn, 4), "point_ens": round(m_en, 4),
    "densenet_ci95": [round(float(np.percentile(s_dn, 2.5)), 4), round(float(np.percentile(s_dn, 97.5)), 4)],
    "convnext_ci95": [round(float(np.percentile(s_cn, 2.5)), 4), round(float(np.percentile(s_cn, 97.5)), 4)],
    "ensemble_ci95": [round(float(np.percentile(s_en, 2.5)), 4), round(float(np.percentile(s_en, 97.5)), 4)],
    "delta_vs_dn_mean": round(float(np.mean(d_dn)), 4),
    "delta_vs_dn_ci95": [round(float(np.percentile(d_dn, 2.5)), 4), round(float(np.percentile(d_dn, 97.5)), 4)],
    "delta_vs_cn_mean": round(float(np.mean(d_cn)), 4),
    "delta_vs_cn_ci95": [round(float(np.percentile(d_cn, 2.5)), 4), round(float(np.percentile(d_cn, 97.5)), 4)],
    "wins_vs_dn": int(sum(x > 0 for x in d_dn)),
    "wins_vs_cn": int(sum(x > 0 for x in d_cn)),
}
with open("./checkpoints/phase7_patient_cluster_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(boot_res, f, indent=2)

npz_mb = os.path.getsize("./checkpoints/nih14_test_predictions.npz") / (1024 * 1024)
print(f"Saved checkpoints/nih14_test_predictions.npz ({npz_mb:.2f} MB)")
print("EXACT BUG-FREE NIH-14 RESULT:", json.dumps(boot_res, indent=2))
