import json, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

NIH_14_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
CLASS_TO_IDX = {c.lower().replace("_", " "): i for i, c in enumerate(NIH_14_CLASSES)}


def encode_multilabel(label_names):
    vec = np.zeros(len(NIH_14_CLASSES), dtype=np.float32)
    for name in (label_names or []):
        key = str(name).strip().lower().replace("_", " ")
        if key in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[key]] = 1.0
    return vec.tolist()


def prepare_nih14_cache(cache_dir="./nih_cxr14_cache", n_train=12000, n_val=2000, n_test=3000):
    os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
    meta_path = os.path.join(cache_dir, "splits_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if len(meta.get("train", [])) >= n_train and len(meta.get("test", [])) >= n_test:
            print(f"[CACHE] Loaded existing NIH-14 cache ({len(meta['train'])} train / {len(meta['val'])} val / {len(meta['test'])} test).", flush=True)
            return meta

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    print(f"[STREAMING NIH-14] Caching {n_train} Train + {n_val} Val + {n_test} Official Test adult X-rays (256x256)...", flush=True)
    meta = {"train": [], "val": [], "test": []}

    # Stream train split -> patient-disjoint split via official fold (fold==4 -> val, fold!=4 -> train)
    ds_tr = load_dataset("timm/nih-chest-xray-14", split="train", streaming=True)
    t0 = time.perf_counter()
    for row in ds_tr:
        fold = int(row.get("fold", 0))
        bucket = "val" if fold == 4 else "train"
        target_cap = n_val if bucket == "val" else n_train
        if len(meta[bucket]) >= target_cap:
            if len(meta["train"]) >= n_train and len(meta["val"]) >= n_val:
                break
            continue
        img_id = row["image_id"]
        fname = f"{img_id}.jpg"
        fpath = os.path.join(cache_dir, "images", fname)
        if not os.path.exists(fpath):
            row["image"].convert("L").resize((256, 256), Image.BILINEAR).save(fpath, quality=92)
        meta[bucket].append({"file": fname, "labels": encode_multilabel(row.get("label_names", []))})
        total_so_far = len(meta["train"]) + len(meta["val"])
        if total_so_far % 2000 == 0:
            print(f"  Cached {len(meta['train'])}/{n_train} train, {len(meta['val'])}/{n_val} val ({time.perf_counter()-t0:.0f}s)...", flush=True)

    # Stream official patient-disjoint test split
    ds_te = load_dataset("timm/nih-chest-xray-14", split="test", streaming=True)
    for row in ds_te:
        if len(meta["test"]) >= n_test:
            break
        img_id = row["image_id"]
        fname = f"{img_id}.jpg"
        fpath = os.path.join(cache_dir, "images", fname)
        if not os.path.exists(fpath):
            row["image"].convert("L").resize((256, 256), Image.BILINEAR).save(fpath, quality=92)
        meta["test"].append({"file": fname, "labels": encode_multilabel(row.get("label_names", []))})
        if len(meta["test"]) % 1000 == 0:
            print(f"  Cached {len(meta['test'])}/{n_test} official test ({time.perf_counter()-t0:.0f}s)...", flush=True)

    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[CACHE COMPLETE] Saved {len(meta['train'])} train / {len(meta['val'])} val / {len(meta['test'])} test in {time.perf_counter()-t0:.1f}s!", flush=True)
    return meta


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


def run_eval_pass(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                probs = torch.sigmoid(logits)
            all_y.append(y.numpy())
            all_p.append(probs.float().cpu().numpy())
    return np.vstack(all_y), np.vstack(all_p)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    device = torch.device("cuda")
    os.makedirs("./checkpoints", exist_ok=True)

    n_train = int(os.environ.get("NIH_TRAIN_COUNT", "12000"))
    n_val   = int(os.environ.get("NIH_VAL_COUNT", "2000"))
    n_test  = int(os.environ.get("NIH_TEST_COUNT", "3000"))
    epochs  = int(os.environ.get("NIH_EPOCHS", "6"))

    print("=" * 88)
    print(f"  PHASE 3A: Real Adult Multi-Label NIH ChestX-ray14 Training ({n_train} Train / {n_val} Val / {n_test} Test)")
    print(f"  Architecture: DenseNet-121 (CheXNet) | GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 88)

    meta = prepare_nih14_cache(n_train=n_train, n_val=n_val, n_test=n_test)
    img_dir = "./nih_cxr14_cache/images"

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.88, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_loader = DataLoader(NIH14Dataset(meta["train"], img_dir, train_tf), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(NIH14Dataset(meta["val"], img_dir, eval_tf), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    te_loader = DataLoader(NIH14Dataset(meta["test"], img_dir, eval_tf), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Compute capped positive class weights to handle medical class imbalance
    y_tr_mat = np.array([r["labels"] for r in meta["train"]], dtype=np.float32)
    pos_counts = np.maximum(y_tr_mat.sum(axis=0), 1.0)
    neg_counts = len(y_tr_mat) - pos_counts
    pos_weights = np.clip(neg_counts / pos_counts, 1.0, 12.0)

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.classifier.in_features, len(NIH_14_CLASSES)))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = 0.0
    best_ckpt = "./checkpoints/medvision_nih14_densenet121.pth"

    print("\n>>> Starting DenseNet-121 Multi-Label Training on RTX 4080...", flush=True)
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        y_va, p_va = run_eval_pass(model, va_loader, device)
        val_aucs = [roc_auc_score(y_va[:, i], p_va[:, i]) for i in range(len(NIH_14_CLASSES)) if y_va[:, i].sum() > 0]
        mean_val_auc = float(np.mean(val_aucs))
        ep_loss = running_loss / len(meta["train"])
        print(f"  Epoch {ep}/{epochs} | Train Weighted BCE: {ep_loss:.4f} | Val Mean AUROC: {mean_val_auc:.4f} ({time.perf_counter()-t_start:.0f}s)", flush=True)
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            torch.save({"model_state_dict": model.state_dict(), "classes": NIH_14_CLASSES, "val_mean_auroc": best_val_auc}, best_ckpt)

    # Load best checkpoint -> Calibrate optimal Youden's J thresholds on Validation -> Evaluate on 3,000 Official Test X-Rays!
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    y_va, p_va = run_eval_pass(model, va_loader, device)
    y_te, p_te = run_eval_pass(model, te_loader, device)

    results = {}
    aucs, sens_list, spec_list, f1_list = [], [], [], []
    print("\n" + "=" * 96)
    print(f"  OFFICIAL PATIENT-DISJOINT NIH CHESTX-RAY14 TEST BENCHMARK (n={len(meta['test'])} Adult Radiographs)")
    print("=" * 96)
    print(f"  {'Pathology':<20} | {'AUROC':>7} | {'Sens (Rec)':>10} | {'Specificity':>11} | {'F1':>7} | {'TP/FN/FP/TN'}")
    print("-" * 96)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0f172a")
    key_plot_classes = {"Effusion", "Cardiomegaly", "Edema", "Pneumothorax", "Emphysema", "Consolidation", "Atelectasis", "Pneumonia"}

    for i, cname in enumerate(NIH_14_CLASSES):
        if y_te[:, i].sum() == 0 or y_va[:, i].sum() == 0:
            continue
        # Calibrate Youden's J threshold strictly on Validation split (zero test leakage!)
        fpr_v, tpr_v, thr_v = roc_curve(y_va[:, i], p_va[:, i])
        best_thr = float(thr_v[int(np.argmax(tpr_v - fpr_v))])

        auc = float(roc_auc_score(y_te[:, i], p_te[:, i]))
        preds_bin = (p_te[:, i] >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te[:, i], preds_bin, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1   = float(f1_score(y_te[:, i], preds_bin, zero_division=0))

        aucs.append(auc); sens_list.append(sens); spec_list.append(spec); f1_list.append(f1)
        results[cname] = {"auroc": round(auc, 4), "sensitivity": round(sens, 4), "specificity": round(spec, 4), "f1": round(f1, 4), "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}
        print(f"  {cname:<20} | {auc:7.4f} | {sens*100:9.1f}% | {spec*100:10.1f}% | {f1*100:6.1f}% | TP={tp} FN={fn} FP={fp} TN={tn}")

        if cname in key_plot_classes:
            fpr_t, tpr_t, _ = roc_curve(y_te[:, i], p_te[:, i])
            ax.plot(fpr_t, tpr_t, linewidth=2.0, label=f"{cname} (AUC={auc:.3f})")

    mean_auc  = float(np.mean(aucs))
    mean_sens = float(np.mean(sens_list))
    mean_spec = float(np.mean(spec_list))
    mean_f1   = float(np.mean(f1_list))

    print("-" * 96)
    print(f"  {'MEAN (14 PATHOLOGIES)':<20} | {mean_auc:7.4f} | {mean_sens*100:9.1f}% | {mean_spec*100:10.1f}% | {mean_f1*100:6.1f}%")
    print("=" * 96)

    # Save 300-DPI ROC Curve Figure
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2)
    ax.set_title(f"MedVision DenseNet-121 — NIH ChestX-ray14 Official Test ROC (Mean AUROC = {mean_auc:.4f})", color="white", fontsize=11, pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#cbd5e1")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=8.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="white")
    roc_path = "./checkpoints/phase3a_nih14_roc_300dpi.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, facecolor="#070b14")

    summary = {
        "dataset": f"timm/nih-chest-xray-14 (Patient-Disjoint: {n_train} Train / {n_val} Val / {n_test} Test)",
        "architecture": "DenseNet-121 (Class-Weighted BCE + Youden's J Calibration)",
        "mean_auroc": round(mean_auc, 4),
        "mean_sensitivity_pct": round(mean_sens * 100, 2),
        "mean_specificity_pct": round(mean_spec * 100, 2),
        "mean_f1_pct": round(mean_f1 * 100, 2),
        "per_class": results,
        "checkpoint": os.path.abspath(best_ckpt),
        "roc_figure": os.path.abspath(roc_path),
    }
    with open("./checkpoints/phase3a_nih14_vision_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved ROC Figure & Report: {os.path.abspath(roc_path)}")
    os.startfile(os.path.abspath(roc_path))

if __name__ == "__main__":
    main()
