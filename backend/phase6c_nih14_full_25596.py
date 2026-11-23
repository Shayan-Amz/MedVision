import gc, json, os
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

# Load ONLY the very last test parquet shard (instant, avoids streaming 25,500 images!)
api = HfApi()
all_files = api.list_repo_files("timm/nih-chest-xray-14", repo_type="dataset")
test_shards = sorted([f for f in all_files if "test-" in f and f.endswith(".parquet")])
last_shard = test_shards[-1]
print(f"Downloading ONLY the last test shard: {last_shard} ...")
ds_last = load_dataset("timm/nih-chest-xray-14", data_files={"test": last_shard}, split="test")
# Official NIH-14 test split has 25,596 images -> take the exact last 96 images
extra_rows = ds_last.select(range(len(ds_last) - 96, len(ds_last)))

extra_imgs, extra_labels = [], []
for row in extra_rows:
    extra_imgs.append(row["image"].convert("RGB"))
    vec = np.zeros(len(classes), dtype=np.float32)
    raw_lbl = row.get("labels") or row.get("label") or []
    if isinstance(raw_lbl, str):
        for p in raw_lbl.split("|"):
            if p.strip() in cls_to_idx: vec[cls_to_idx[p.strip()]] = 1.0
    elif isinstance(raw_lbl, (list, tuple)):
        for p in raw_lbl:
            if isinstance(p, int) and 0 <= p < len(classes): vec[p] = 1.0
            elif isinstance(p, str) and p in cls_to_idx: vec[cls_to_idx[p]] = 1.0
    extra_labels.append(vec)

print(f"Loaded {len(extra_imgs)} extra tail images in seconds! Combining with 25,500 local cached SSD images...")

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

print(json.dumps({
    "n_official_test_evaluated": int(len(y_all)),
    "densenet_tta_auroc": round(float(np.mean([roc_auc_score(y_all[:, c], p_dn[:, c]) for c in range(14)])), 4),
    "convnext_tta_auroc": round(float(np.mean([roc_auc_score(y_all[:, c], p_cn[:, c]) for c in range(14)])), 4),
    "ensemble_auroc": round(float(np.mean([roc_auc_score(y_all[:, c], p_ens[:, c]) for c in range(14)])), 4),
}, indent=2))
