import os, json, threading
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models
import torchxrayvision as xrv
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

NIH14_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

STAT_PATHOLOGIES = {"Pneumothorax", "Edema", "Effusion", "Consolidation"}

_VLOCK = threading.Lock()
_DENSENET = None
_CONVNEXT = None
_SEG_MODEL = None
_META = None
_DEVICE = None

_TF_224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_TF_256 = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_vision_suite():
    global _DENSENET, _CONVNEXT, _SEG_MODEL, _META, _DEVICE
    if _DENSENET is not None:
        return _DENSENET, _CONVNEXT, _SEG_MODEL, _META, _DEVICE

    with _VLOCK:
        if _DENSENET is not None:
            return _DENSENET, _CONVNEXT, _SEG_MODEL, _META, _DEVICE

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Ensemble Meta
        meta_path = "./checkpoints/medvision_nih14_ensemble_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                _META = json.load(f)
        else:
            _META = {
                "convnext_weight": 0.45,
                "densenet_weight": 0.55,
                "thresholds": {c: 0.50 for c in NIH14_CLASSES},
            }

        # 1. DenseNet-121
        dn_ckpt = torch.load("./checkpoints/medvision_nih14_densenet121.pth", map_location=_DEVICE, weights_only=False)
        dn_state = dn_ckpt["model_state_dict"] if "model_state_dict" in dn_ckpt else dn_ckpt
        dn = tv_models.densenet121(weights=None)
        in_dn = dn.classifier.in_features
        if "classifier.1.weight" in dn_state:
            dn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dn, len(NIH14_CLASSES)))
        else:
            dn.classifier = nn.Linear(in_dn, len(NIH14_CLASSES))
        dn.load_state_dict(dn_state)
        _DENSENET = dn.to(_DEVICE).eval()

        # 2. ConvNeXt-Small
        cn_path = "./checkpoints/medvision_nih14_convnext_small.pth"
        if os.path.exists(cn_path):
            cn_ckpt = torch.load(cn_path, map_location=_DEVICE, weights_only=False)
            cn = tv_models.convnext_small(weights=None)
            in_cn = cn.classifier[2].in_features
            cn.classifier[2] = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_cn, len(NIH14_CLASSES)))
            cn.load_state_dict(cn_ckpt["model_state_dict"])
            _CONVNEXT = cn.to(_DEVICE).eval()
        else:
            _CONVNEXT = None

        # 3. PSPNet Anatomical Segmenter
        _SEG_MODEL = xrv.baseline_models.chestx_det.PSPNet().to(_DEVICE).eval()
        return _DENSENET, _CONVNEXT, _SEG_MODEL, _META, _DEVICE


def check_ood_and_segment(pil_rgb: Image.Image) -> Dict[str, Any]:
    """Checks for non-X-ray OOD images and computes PSPNet lung/heart masks & exact CTR."""
    _, _, seg_model, _, device = load_vision_suite()

    # Check color saturation (natural photos have high R-G/G-B divergence)
    rgb_np = np.array(pil_rgb.resize((256, 256)), dtype=np.float32)
    color_diff = float(np.mean(np.abs(rgb_np[:, :, 0] - rgb_np[:, :, 1]) + np.abs(rgb_np[:, :, 1] - rgb_np[:, :, 2])))
    is_color_ood = color_diff > 28.0

    gray = pil_rgb.convert("L").resize((512, 512), Image.BILINEAR)
    arr = xrv.datasets.normalize(np.array(gray, dtype=np.float32), 255)
    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = seg_model(inp)[0]
        probs = torch.sigmoid(out) if (out.min() < 0 or out.max() > 1) else out

    idx_l = seg_model.targets.index("Left Lung")
    idx_r = seg_model.targets.index("Right Lung")
    idx_h = seg_model.targets.index("Heart")

    l_mask = (probs[idx_l] > 0.5).cpu().numpy()
    r_mask = (probs[idx_r] > 0.5).cpu().numpy()
    h_mask = (probs[idx_h] > 0.5).cpu().numpy()
    lung_union = np.logical_or(l_mask, r_mask)

    lung_coverage = float(lung_union.mean())
    is_lung_ood = lung_coverage < 0.08

    # Compute Exact Cardiothoracic Ratio (CTR)
    ctr_val = None
    if lung_union.any() and h_mask.any():
        lung_cols = np.where(lung_union.any(axis=0))[0]
        heart_cols = np.where(h_mask.any(axis=0))[0]
        thorax_w = float(lung_cols[-1] - lung_cols[0])
        heart_w = float(heart_cols[-1] - heart_cols[0])
        if thorax_w > 20:
            ctr_val = round(heart_w / thorax_w, 3)

    return {
        "is_ood": bool(is_color_ood or is_lung_ood),
        "ood_reason": (
            f"Non-radiograph detected (color divergence={color_diff:.1f}, lung mask coverage={lung_coverage*100:.1f}%)."
            if (is_color_ood or is_lung_ood) else ""
        ),
        "lung_coverage_pct": round(lung_coverage * 100, 1),
        "ctr": ctr_val,
        "lung_mask": lung_union,
        "heart_mask": h_mask,
    }


def analyze_chest_xray_ensemble(pil_rgb: Image.Image, operating_mode: str = "triage") -> Dict[str, Any]:
    """
    Runs OOD check -> PSPNet Segmentation/CTR -> DenseNet-121 + ConvNeXt-Small (2-view TTA) ->
    Finding-Aware Lung-Gated Grad-CAM++ overlay.
    operating_mode: 'triage' (High-Sensitivity Youden's J) or 'balanced' (High-Specificity, lower alarm fatigue)
    """
    dn, cn, _, meta, device = load_vision_suite()
    seg_info = check_ood_and_segment(pil_rgb)
    if seg_info["is_ood"]:
        return {"ood_rejected": True, "ood_reason": seg_info["ood_reason"], "lung_coverage_pct": seg_info["lung_coverage_pct"]}

    x224 = _TF_224(pil_rgb).unsqueeze(0).to(device)
    x256 = _TF_256(pil_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p_dn = 0.5 * (torch.sigmoid(dn(x224).float()) + torch.sigmoid(dn(torch.flip(x224, dims=[3])).float()))
            if cn is not None:
                p_cn = 0.5 * (torch.sigmoid(cn(x256).float()) + torch.sigmoid(cn(torch.flip(x256, dims=[3])).float()))
                w_cn = float(meta.get("convnext_weight", 0.45))
                w_dn = float(meta.get("densenet_weight", 0.55))
                probs = (w_cn * p_cn + w_dn * p_dn)[0].cpu().numpy()
            else:
                probs = p_dn[0].cpu().numpy()

    thresholds = meta.get("thresholds", {c: 0.50 for c in NIH14_CLASSES})
    shift = 0.15 if operating_mode == "balanced" else 0.0

    findings = []
    for i, cname in enumerate(NIH14_CLASSES):
        p = float(probs[i])
        thr = min(0.92, float(thresholds.get(cname, 0.50)) + shift)
        findings.append({"name": cname, "prob": round(p, 4), "threshold": round(thr, 3), "positive": bool(p >= thr)})

    findings.sort(key=lambda d: d["prob"], reverse=True)
    primary = findings[0]["name"]
    primary_idx = NIH14_CLASSES.index(primary)
    top_conf = findings[0]["prob"]

    # Urgency stratification
    pos_names = {f["name"] for f in findings if f["positive"]}
    if (pos_names & STAT_PATHOLOGIES) and top_conf >= 0.72:
        urgency = "STAT"
    elif pos_names and top_conf >= 0.62:
        urgency = "HIGH"
    else:
        urgency = "ROUTINE"

    # Finding-Aware Lung-Gated Grad-CAM++ on DenseNet-121
    with _VLOCK:
        dn.zero_grad()
        feats = F.relu(dn.features(x224), inplace=False)
        feats.retain_grad()
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        logits = dn.classifier(pooled)[0]
        logits[primary_idx].backward()

        A = feats.detach()[0]
        G = feats.grad.detach()[0]
        weights = F.relu(G).mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * A).sum(dim=0))
        cam_512 = F.interpolate(cam[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        if cam_512.max() > 0:
            cam_512 = cam_512 / cam_512.max()

    # Keep heart region if Cardiomegaly is primary OR CTR > 0.50
    ctr_val = seg_info["ctr"]
    keep_heart = (primary == "Cardiomegaly") or (ctr_val is not None and ctr_val > 0.50)
    gate_mask = np.logical_or(seg_info["lung_mask"], seg_info["heart_mask"]) if keep_heart else np.logical_and(seg_info["lung_mask"], ~seg_info["heart_mask"])

    cam_gated = cam_512 * gate_mask
    if cam_gated.max() > 0:
        cam_gated = cam_gated / cam_gated.max()
    else:
        cam_gated = cam_512

    base_512 = np.array(pil_rgb.resize((512, 512)), dtype=np.float32) / 255.0
    if urgency == "ROUTINE":
        # Normal/borderline scan: subtle emerald lung-mask tint only, no false red lesion hotspot
        tint = np.zeros_like(base_512)
        tint[:, :, 1] = 0.75
        tint[:, :, 2] = 0.65
        alpha = seg_info["lung_mask"].astype(np.float32)[:, :, None] * 0.12
        overlay = np.clip(base_512 * (1.0 - alpha) + tint * alpha, 0.0, 1.0)
    else:
        # Focal lesion-only Grad-CAM++ (suppress low-activation blue wash across healthy lung)
        heat_rgb = cm.inferno(cam_gated)[:, :, :3]
        focal = np.clip((cam_gated - 0.42) / 0.58, 0.0, 1.0)
        alpha = (focal ** 1.2)[:, :, None] * 0.62
        overlay = np.clip(base_512 * (1.0 - alpha) + heat_rgb * alpha, 0.0, 1.0)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    return {
        "ood_rejected": False,
        "primary_finding": primary,
        "top_confidence": top_conf,
        "urgency": urgency,
        "operating_mode": operating_mode,
        "ctr": ctr_val,
        "heart_preserved_in_cam": keep_heart,
        "lung_coverage_pct": seg_info["lung_coverage_pct"],
        "findings": findings,
        "top3": [(f["name"], f["prob"]) for f in findings[:3]],
        "gradcam_image": overlay_pil,
    }
