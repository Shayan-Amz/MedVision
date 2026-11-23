import os, json, glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models
import torchxrayvision as xrv
from huggingface_hub import snapshot_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 88)
print(f"  PHASE 3C (Part 2): Grad-CAM++ Radiologist BBox Localization Eval")
print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 88)

seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
idx_left = seg_model.targets.index("Left Lung")
idx_right = seg_model.targets.index("Right Lung")
idx_heart = seg_model.targets.index("Heart")

bbox_dir = snapshot_download(
    repo_id="Zackboulhand/xray-chest-cases",
    repo_type="dataset"
)

ckpt_path = "./checkpoints/medvision_nih14_densenet121.pth"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
classes = ckpt.get("classes", [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
])
cls_lower = {c.lower().replace("_", " "): i for i, c in enumerate(classes)}

densenet = tv_models.densenet121(weights=None)
in_feats = densenet.classifier.in_features
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
if "classifier.1.weight" in state:
    densenet.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_feats, len(classes))
    )
else:
    densenet.classifier = torch.nn.Linear(in_feats, len(classes))
densenet.load_state_dict(state)
densenet = densenet.to(device).eval()

norm_tf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_boxes(meta):
    boxes = []
    raw = meta.get("boxes") or meta.get("bbox") or meta.get("box") or meta.get("annotations") or []
    if isinstance(raw, dict):
        raw = [raw]
    for b in raw:
        if isinstance(b, dict):
            if all(k in b for k in ("x", "y", "w", "h")):
                boxes.append((float(b["x"]), float(b["y"]), float(b["x"] + b["w"]), float(b["y"] + b["h"])))
            elif all(k in b for k in ("x1", "y1", "x2", "y2")):
                boxes.append((float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])))
            elif all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                boxes.append((float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])))
            elif "box" in b and isinstance(b["box"], (list, tuple)) and len(b["box"]) == 4:
                x, y, w, h = map(float, b["box"])
                boxes.append((x, y, x + w, y + h))
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            x, y, w, h = map(float, b)
            boxes.append((x, y, x + w, y + h))
    return boxes

def map_finding(meta):
    txt = str(meta.get("finding") or meta.get("label") or meta.get("disease") or "").lower().replace("_", " ")
    if "effusion" in txt:
        return cls_lower.get("effusion")
    for k, idx in cls_lower.items():
        if k in txt:
            return idx
    return None

data_jsons = sorted(glob.glob(os.path.join(bbox_dir, "**", "data.json"), recursive=True))
pointing_hits_raw, pointing_hits_gated = [], []
energy_in_box_raw, energy_in_box_gated = [], []
per_class_hits = {}

for jpath in data_jsons:
    cdir = os.path.dirname(jpath)
    ipath = os.path.join(cdir, "image.png")
    if not os.path.exists(ipath):
        continue
    with open(jpath, "r", encoding="utf-8") as f:
        meta = json.load(f)

    c_idx = map_finding(meta)
    boxes = extract_boxes(meta)
    if c_idx is None or not boxes:
        continue

    pil_rgb = Image.open(ipath).convert("RGB")
    W, H = pil_rgb.size
    inp = norm_tf(pil_rgb).unsqueeze(0).to(device)

    densenet.zero_grad()
    feats = F.relu(densenet.features(inp), inplace=False)
    feats.retain_grad()
    pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
    logits = densenet.classifier(pooled)[0]
    logits[c_idx].backward()

    A = feats.detach()[0]
    G = feats.grad.detach()[0]
    weights = F.relu(G).mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * A).sum(dim=0))
    cam_up = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    if cam_up.max() > 0:
        cam_up = cam_up / cam_up.max()

    with torch.no_grad():
        gray = pil_rgb.convert("L").resize((512, 512), Image.BILINEAR)
        arr_seg = xrv.datasets.normalize(np.array(gray, dtype=np.float32), 255)
        seg_out = seg_model(torch.from_numpy(arr_seg).unsqueeze(0).unsqueeze(0).to(device))[0]
        seg_probs = torch.sigmoid(seg_out) if (seg_out.min() < 0 or seg_out.max() > 1) else seg_out
        lung_mask = torch.logical_or(seg_probs[idx_left] > 0.5, seg_probs[idx_right] > 0.5)
        if classes[c_idx] != "Cardiomegaly":
            lung_mask = torch.logical_and(lung_mask, ~(seg_probs[idx_heart] > 0.5))
        lung_mask_full = F.interpolate(lung_mask[None, None].float(), size=(H, W), mode="nearest")[0, 0].cpu().numpy() > 0.5

    cam_gated = cam_up * lung_mask_full
    if cam_gated.max() > 0:
        cam_gated = cam_gated / cam_gated.max()
    else:
        cam_gated = cam_up

    gt_mask = np.zeros((H, W), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        gt_mask[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))] = True

    py_r, px_r = np.unravel_index(np.argmax(cam_up), cam_up.shape)
    py_g, px_g = np.unravel_index(np.argmax(cam_gated), cam_gated.shape)

    hit_r = bool(gt_mask[py_r, px_r])
    hit_g = bool(gt_mask[py_g, px_g])
    pointing_hits_raw.append(hit_r)
    pointing_hits_gated.append(hit_g)

    e_r = float(cam_up[gt_mask].sum() / (cam_up.sum() + 1e-8))
    e_g = float(cam_gated[gt_mask].sum() / (cam_gated.sum() + 1e-8))
    energy_in_box_raw.append(e_r)
    energy_in_box_gated.append(e_g)

    cname = classes[c_idx]
    per_class_hits.setdefault(cname, []).append(hit_g)

print(f"\nEvaluated {len(pointing_hits_gated)} Radiologist BBox Cases (Weakly-Supervised Localization):")
if len(pointing_hits_gated) == 0 and data_jsons:
    with open(data_jsons[0], "r", encoding="utf-8") as f:
        print("  DEBUG sample data.json:", json.load(f))
else:
    print(f"  - Raw Grad-CAM++ Pointing Game Acc        : {np.mean(pointing_hits_raw)*100:.2f}% | BBox Energy: {np.mean(energy_in_box_raw)*100:.2f}%")
    print(f"  - Lung-Gated Grad-CAM++ Pointing Game Acc : {np.mean(pointing_hits_gated)*100:.2f}% | BBox Energy: {np.mean(energy_in_box_gated)*100:.2f}%")
    print("\n  Per-Finding Pointing Game Accuracy (Lung-Gated):")
    for k, vals in sorted(per_class_hits.items()):
        print(f"    - {k:16s}: {np.mean(vals)*100:6.2f}% ({sum(vals)}/{len(vals)})")
