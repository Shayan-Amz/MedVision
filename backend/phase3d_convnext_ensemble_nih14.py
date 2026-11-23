import os, json, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIH14_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

class NIH14Dataset(Dataset):
    def __init__(self, records, img_dir, transform):
        self.records = records
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.img_dir, rec["file"])).convert("RGB")
        return self.transform(img), torch.tensor(rec["labels"], dtype=torch.float32)

def compute_auroc_per_class(y_true, y_prob):
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return np.array(aucs, dtype=np.float64)

@torch.no_grad()
def predict_probs(model, loader, device, tta_hflip=True):
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p1 = torch.sigmoid(model(x).float())
            if tta_hflip:
                p2 = torch.sigmoid(model(torch.flip(x, dims=[3])).float())
                p = 0.5 * (p1 + p2)
            else:
                p = p1
        all_probs.append(p.cpu().numpy())
        all_targets.append(y.numpy())
    return np.vstack(all_probs), np.vstack(all_targets)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    cache_dir = "./nih_cxr14_cache"
    img_dir = os.path.join(cache_dir, "images")
    with open(os.path.join(cache_dir, "splits_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta.get("classes", NIH14_CLASSES)
    n_classes = len(classes)

    print("=" * 90)
    print("  PHASE 3D: ConvNeXt-Small (256x256) Training + DenseNet-121 Ensemble on NIH ChestX-ray14")
    print(f"  Train: {len(meta['train'])} | Val: {len(meta['val'])} | Official Test: {len(meta['test'])}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90)

    train_tf = T.Compose([
        T.RandomResizedCrop(256, scale=(0.85, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(7),
        T.ColorJitter(brightness=0.12, contrast=0.12),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_256_tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_224_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_loader = DataLoader(NIH14Dataset(meta["train"], img_dir, train_tf), batch_size=48, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_loader_256 = DataLoader(NIH14Dataset(meta["val"], img_dir, eval_256_tf), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    te_loader_256 = DataLoader(NIH14Dataset(meta["test"], img_dir, eval_256_tf), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Capped positive class weights strictly from Train split
    y_tr = np.array([r["labels"] for r in meta["train"]], dtype=np.float32)
    pos_counts = np.maximum(y_tr.sum(axis=0), 1.0)
    neg_counts = np.maximum(len(y_tr) - pos_counts, 1.0)
    pos_weights = np.clip(neg_counts / pos_counts, 1.0, 12.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=device))

    # Build ConvNeXt-Small
    convnext = tv_models.convnext_small(weights=tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    in_feats = convnext.classifier[2].in_features
    convnext.classifier[2] = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, n_classes))
    convnext = convnext.to(device)

    epochs = 8
    optimizer = torch.optim.AdamW(convnext.parameters(), lr=1.5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("./checkpoints", exist_ok=True)
    conv_ckpt_path = "./checkpoints/medvision_nih14_convnext_small.pth"
    best_val_auc = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        convnext.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(tr_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = convnext(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(convnext.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item()
            if step % 300 == 0 or step == len(tr_loader):
                print(f"  [Epoch {ep}/{epochs}] Step {step}/{len(tr_loader)} | Loss: {running_loss/step:.4f}")

        scheduler.step()
        val_probs, val_y = predict_probs(convnext, va_loader_256, device, tta_hflip=False)
        val_aucs = compute_auroc_per_class(val_y, val_probs)
        val_mean_auc = float(np.nanmean(val_aucs))
        elapsed = time.time() - t0
        print(f"  --> Epoch {ep} Done in {elapsed/60:.1f}m | Val Mean AUROC: {val_mean_auc:.4f} (Best: {max(best_val_auc, val_mean_auc):.4f})")

        if val_mean_auc > best_val_auc:
            best_val_auc = val_mean_auc
            torch.save({
                "model_state_dict": convnext.state_dict(),
                "classes": classes,
                "val_mean_auroc": val_mean_auc,
                "arch": "convnext_small",
                "img_size": 256,
            }, conv_ckpt_path)
            print(f"      [SAVED] New Best ConvNeXt-Small -> {conv_ckpt_path}")

    # -------------------------------------------------------------------------
    # Load Best ConvNeXt-Small & Existing DenseNet-121 for Leak-Free Ensemble
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  EVALUATING ENSEMBLE (Optimal Blend Weight Tuned Strictly on Val -> Tested on Official Test)")
    print("=" * 90)

    conv_ckpt = torch.load(conv_ckpt_path, map_location=device, weights_only=False)
    convnext.load_state_dict(conv_ckpt["model_state_dict"])

    dn_ckpt = torch.load("./checkpoints/medvision_nih14_densenet121.pth", map_location=device, weights_only=False)
    densenet = tv_models.densenet121(weights=None)
    dn_in = densenet.classifier.in_features
    dn_state = dn_ckpt["model_state_dict"] if "model_state_dict" in dn_ckpt else dn_ckpt
    if "classifier.1.weight" in dn_state:
        densenet.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(dn_in, n_classes))
    else:
        densenet.classifier = nn.Linear(dn_in, n_classes)
    densenet.load_state_dict(dn_state)
    densenet = densenet.to(device).eval()

    va_loader_224 = DataLoader(NIH14Dataset(meta["val"], img_dir, eval_224_tf), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    te_loader_224 = DataLoader(NIH14Dataset(meta["test"], img_dir, eval_224_tf), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    p_val_conv, y_val = predict_probs(convnext, va_loader_256, device, tta_hflip=True)
    p_val_dn, _ = predict_probs(densenet, va_loader_224, device, tta_hflip=True)

    best_w, best_ens_val = 0.5, 0.0
    for w in np.linspace(0.0, 1.0, 21):
        p_mix = w * p_val_conv + (1.0 - w) * p_val_dn
        m_auc = float(np.nanmean(compute_auroc_per_class(y_val, p_mix)))
        if m_auc > best_ens_val:
            best_ens_val = m_auc
            best_w = float(w)

    print(f"  [Val] DenseNet-121 (TTA) AUROC : {np.nanmean(compute_auroc_per_class(y_val, p_val_dn)):.4f}")
    print(f"  [Val] ConvNeXt-Small (TTA) AUROC: {np.nanmean(compute_auroc_per_class(y_val, p_val_conv)):.4f}")
    print(f"  [Val] Optimal Ensemble Weight  : {best_w:.2f} * ConvNeXt + {1-best_w:.2f} * DenseNet -> Val AUROC: {best_ens_val:.4f}")

    # Calibrate Youden's J thresholds on Val for the Ensemble
    p_val_ens = best_w * p_val_conv + (1.0 - best_w) * p_val_dn
    ens_thresholds = []
    for c in range(n_classes):
        fpr, tpr, thr = roc_curve(y_val[:, c], p_val_ens[:, c])
        j_idx = np.argmax(tpr - fpr)
        ens_thresholds.append(float(thr[j_idx]))

    # Final Blind Evaluation on 25,500 Official Test
    p_te_conv, y_te = predict_probs(convnext, te_loader_256, device, tta_hflip=True)
    p_te_dn, _ = predict_probs(densenet, te_loader_224, device, tta_hflip=True)
    p_te_ens = best_w * p_te_conv + (1.0 - best_w) * p_te_dn

    aucs_dn = compute_auroc_per_class(y_te, p_te_dn)
    aucs_conv = compute_auroc_per_class(y_te, p_te_conv)
    aucs_ens = compute_auroc_per_class(y_te, p_te_ens)

    print("\n  --- OFFICIAL 25,500 TEST SET COMPARISON ---")
    print(f"  {'Pathology':20s} | {'DenseNet-121':12s} | {'ConvNeXt-S':12s} | {'Ensemble':12s} | {'Sens':7s} | {'Spec':7s}")
    print("  " + "-" * 80)
    sens_list, spec_list = [], []
    for c, cname in enumerate(classes):
        pred_bin = (p_te_ens[:, c] >= ens_thresholds[c]).astype(int)
        yt = y_te[:, c]
        tp = ((pred_bin == 1) & (yt == 1)).sum()
        fn = ((pred_bin == 0) & (yt == 1)).sum()
        tn = ((pred_bin == 0) & (yt == 0)).sum()
        fp = ((pred_bin == 1) & (yt == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        sens_list.append(sens); spec_list.append(spec)
        print(f"  {cname:20s} | {aucs_dn[c]:12.4f} | {aucs_conv[c]:12.4f} | {aucs_ens[c]:12.4f} | {sens*100:6.1f}% | {spec*100:6.1f}%")

    print("  " + "-" * 80)
    print(f"  {'MEAN (14 Classes)':20s} | {np.nanmean(aucs_dn):12.4f} | {np.nanmean(aucs_conv):12.4f} | {np.nanmean(aucs_ens):12.4f} | {np.mean(sens_list)*100:6.1f}% | {np.mean(spec_list)*100:6.1f}%")

    # Save Ensemble Metadata
    with open("./checkpoints/medvision_nih14_ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "convnext_weight": best_w,
            "densenet_weight": 1.0 - best_w,
            "thresholds": dict(zip(classes, ens_thresholds)),
            "test_mean_auroc_densenet": float(np.nanmean(aucs_dn)),
            "test_mean_auroc_convnext": float(np.nanmean(aucs_conv)),
            "test_mean_auroc_ensemble": float(np.nanmean(aucs_ens)),
        }, f, indent=2)

    # Save 300 DPI ROC Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    key_classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema"]
    for cname in key_classes:
        idx = classes.index(cname)
        fpr, tpr, _ = roc_curve(y_te[:, idx], p_te_ens[:, idx])
        ax.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={aucs_ens[idx]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#64748b", lw=1)
    ax.set_title(f"MedVision Ensemble (DenseNet-121 + ConvNeXt-Small) — NIH-14 Official Test (Mean AUROC = {np.nanmean(aucs_ens):.4f})", fontsize=11)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("./checkpoints/phase3d_ensemble_roc_300dpi.png", dpi=300)
    print("\n  [SAVED] 300 DPI Ensemble ROC -> ./checkpoints/phase3d_ensemble_roc_300dpi.png")

if __name__ == "__main__":
    main()
