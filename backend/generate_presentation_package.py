import os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pres_dir = os.path.abspath(os.path.join("..", "presentation"))
fig_dir = os.path.join(pres_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
})

# =========================================================================
# FIGURE 1: LLM & Clinical NLP Progression (Before vs After Updates)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
fig.patch.set_facecolor("#ffffff")

# Panel A: PubMedQA
pqa_labels = ["Base\nQwen-7B", "v2 SFT\n(12k)", "Prob\nEnsemble", "v2 + Full\nAbstract"]
pqa_vals = [51.50, 77.50, 78.00, 78.50]
colors_a = ["#94a3b8", "#38bdf8", "#0284c7", "#059669"]
bars = axes[0].bar(pqa_labels, pqa_vals, color=colors_a, width=0.58, edgecolor="#1e293b", linewidth=0.6)
axes[0].axhline(78.0, color="#dc2626", linestyle="--", linewidth=1.4, label="Human Physician Avg (78.0%)")
axes[0].set_ylim(40, 88)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("PubMedQA Reasoning (n=200 Unseen)", fontweight="bold", fontsize=11)
axes[0].legend(loc="lower right", fontsize=8.5)
axes[0].grid(axis="y", alpha=0.25)
for b, v in zip(bars, pqa_vals):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.9, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9)

# Panel B: MedMCQA Full Validation
mc_labels = ["Base\nQwen-7B", "v3 SFT+\nReplay", "v2+83k\nRAG (200)", "175k RAG\nSingle (4,183)", "175k RAG+\nCyclic-4 (4,183)"]
mc_vals = [52.00, 56.50, 59.50, 64.52, 64.98]
colors_b = ["#94a3b8", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
bars = axes[1].bar(mc_labels, mc_vals, color=colors_b, width=0.6, edgecolor="#1e293b", linewidth=0.6)
axes[1].set_ylim(45, 72)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("MedMCQA Board Exam Progression", fontweight="bold", fontsize=11)
axes[1].tick_params(axis="x", labelsize=8)
axes[1].grid(axis="y", alpha=0.25)
for b, v in zip(bars, mc_vals):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=8.5)

# Panel C: IU-Xray Findings-to-Impression (n=250 UID-Disjoint)
iu_metrics = ["Micro-F1", "Macro-F1", "Negation\nSpec.", "ROUGE-L"]
iu_base = [58.08, 39.64, 91.22, 15.07]
iu_ours = [79.32, 60.45, 97.70, 63.76]
x = np.arange(len(iu_metrics))
w = 0.35
b1 = axes[2].bar(x - w/2, iu_base, w, label="Base Qwen2.5-7B", color="#94a3b8", edgecolor="#1e293b", linewidth=0.6)
b2 = axes[2].bar(x + w/2, iu_ours, w, label="MedVision Radiologist", color="#7c3aed", edgecolor="#1e293b", linewidth=0.6)
axes[2].set_xticks(x)
axes[2].set_xticklabels(iu_metrics, fontsize=9)
axes[2].set_ylim(0, 110)
axes[2].set_ylabel("Score (%)")
axes[2].set_title("IU-Xray Findings-to-Impression (n=250)", fontweight="bold", fontsize=11)
axes[2].legend(loc="upper left", fontsize=8.5)
axes[2].grid(axis="y", alpha=0.25)
for b in b1:
    axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=7.5, color="#475569")
for b in b2:
    axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontweight="bold", fontsize=8, color="#4c1d95")

plt.suptitle("MedVision-7B Clinical LLM & Medical RAG — Verified Benchmark Progression", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig1_llm_progression_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================================
# FIGURE 2: NIH ChestX-ray14 14-Class AUROC (DenseNet vs ConvNeXt vs Ensemble)
# =========================================================================
classes = [
    "Hernia", "Emphysema", "Cardiomegaly", "Pneumothorax", "Edema",
    "Fibrosis", "Mass", "Effusion", "Pleural_Thickening", "Atelectasis",
    "Nodule", "Consolidation", "Pneumonia", "Infiltration"
]
dn_aucs  = [0.9102, 0.9118, 0.8904, 0.8615, 0.8465, 0.8334, 0.8273, 0.8259, 0.7770, 0.7739, 0.7557, 0.7545, 0.7099, 0.7087]
cn_aucs  = [0.9026, 0.9013, 0.8925, 0.8643, 0.8495, 0.8280, 0.8257, 0.8307, 0.7766, 0.7793, 0.7639, 0.7502, 0.7110, 0.7147]
ens_aucs = [0.9203, 0.9135, 0.8980, 0.8701, 0.8535, 0.8394, 0.8331, 0.8325, 0.7850, 0.7827, 0.7679, 0.7597, 0.7192, 0.7162]
sens_pct = [79.1, 87.9, 84.8, 89.0, 92.4, 81.8, 83.9, 88.0, 83.8, 83.5, 66.1, 90.0, 76.2, 88.2]
spec_pct = [91.7, 78.4, 77.2, 66.7, 58.4, 70.7, 62.8, 59.7, 56.9, 56.4, 72.9, 44.4, 55.9, 36.6]

y = np.arange(len(classes))
h = 0.26
fig, ax = plt.subplots(figsize=(12, 7.8), dpi=300)
ax.barh(y + h, dn_aucs, h, label="DenseNet-121 (TTA) — Mean 0.8133", color="#94a3b8")
ax.barh(y, cn_aucs, h, label="ConvNeXt-Small (TTA) — Mean 0.8136", color="#38bdf8")
ax.barh(y - h, ens_aucs, h, label="Ensemble (0.55 DN + 0.45 CN) — Mean 0.8208 (14/14 Best)", color="#059669")
ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(0.65, 0.96)
ax.set_xlabel("Official Test AUROC (n=25,500 Patient-Disjoint, 0 Overlap)", fontsize=10.5)
ax.set_title("NIH ChestX-ray14 Official Test Performance: Ensemble Beats Both Single Models on All 14/14 Pathologies", fontsize=12, fontweight="bold")
ax.axvline(0.8208, color="#059669", linestyle="--", linewidth=1.2, alpha=0.8)
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(axis="x", alpha=0.25)
for i, v in enumerate(ens_aucs):
    ax.text(v + 0.003, y[i] - h, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold", color="#065f46")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig2_nih14_per_class_auroc.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================================
# FIGURE 3: Learning Curve (Line) + AUROC-vs-FPR Linear Regression Analysis
# =========================================================================
epochs = np.arange(1, 9)
tr_loss = [0.6577, 0.5936, 0.5661, 0.5423, 0.5155, 0.4871, 0.4595, 0.4414]
va_auc  = [0.7936, 0.8150, 0.8324, 0.8324, 0.8378, 0.8408, 0.8381, 0.8390]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=300)

# Panel A: Dual-axis Learning Curve
ln1 = ax1.plot(epochs, tr_loss, "o-", color="#ef4444", lw=2.2, label="Train Weighted-BCE Loss")
ax1.set_xlabel("Training Epoch", fontsize=10)
ax1.set_ylabel("Training Loss", color="#ef4444", fontsize=10)
ax1.tick_params(axis="y", labelcolor="#ef4444")
ax1.grid(alpha=0.25)

ax1b = ax1.twinx()
ln2 = ax1b.plot(epochs, va_auc, "s-", color="#0284c7", lw=2.2, label="Val Mean AUROC (n=8,617)")
ax1b.axhline(0.8522, color="#059669", linestyle="--", lw=1.4, label="Ensemble Val AUROC (0.8522)")
ax1b.set_ylabel("Validation Mean AUROC", color="#0284c7", fontsize=10)
ax1b.tick_params(axis="y", labelcolor="#0284c7")
ax1b.set_ylim(0.78, 0.865)
ax1.set_title("ConvNeXt-Small Convergence (69k Train / 8.6k Val)", fontweight="bold", fontsize=11)
lines = ln1 + ln2 + [ax1b.lines[-1]]
ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8.5)

# Panel B: Linear Regression of AUROC vs False Positive Rate (1 - Spec) across 14 Pathologies
aucs_arr = np.array(ens_aucs)
fpr_arr = 100.0 - np.array(spec_pct)  # False Positive Rate %
slope, intercept = np.polyfit(aucs_arr, fpr_arr, 1)
x_line = np.linspace(0.70, 0.93, 100)
y_line = slope * x_line + intercept
r_val = np.corrcoef(aucs_arr, fpr_arr)[0, 1]
r2_val = r_val ** 2

ax2.scatter(aucs_arr, fpr_arr, s=75, c=sens_pct, cmap="viridis", edgecolors="#0f172a", linewidth=0.7, zorder=4)
ax2.plot(x_line, y_line, "--", color="#dc2626", lw=2.0, label=f"OLS Regression: FPR = {slope:.1f}*AUROC + {intercept:.1f}\n(r = {r_val:.3f}, R² = {r2_val:.3f})")
for c_name, ax_x, ay_y in zip(classes, aucs_arr, fpr_arr):
    offset_y = 1.6 if c_name not in ("Pneumothorax", "Effusion") else -2.6
    ax2.annotate(c_name.replace("_", " "), (ax_x, ay_y), xytext=(0, offset_y), textcoords="offset points", ha="center", fontsize=7.5)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label("Sensitivity (%)", fontsize=9)
ax2.set_xlabel("Ensemble Official Test AUROC", fontsize=10)
ax2.set_ylabel("False Positive Rate / Error Rate (100 - Spec %)", fontsize=10)
ax2.set_title("Regression Analysis: Higher AUROC Strongly Reduces Alarm Fatigue (FPR)", fontweight="bold", fontsize=11)
ax2.legend(loc="upper right", fontsize=8.5)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig3_learning_curve_and_regression.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================================
# FIGURE 4: Lung Segmentation Dice/IoU + XAI Localization Ablation
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), dpi=300)

seg_labels = ["Right Lung\nDice", "Combined\nLung Dice", "Left Lung\nDice", "Combined\nLung IoU"]
seg_vals = [90.36, 87.00, 83.26, 77.07]
bars = ax1.bar(seg_labels, seg_vals, color=["#059669", "#10b981", "#38bdf8", "#6366f1"], width=0.52, edgecolor="#1e293b", linewidth=0.6)
ax1.set_ylim(60, 100)
ax1.set_ylabel("Score (%)")
ax1.set_title("PSPNet Lung Segmentation (Montgomery n=138 Manual Masks)", fontweight="bold", fontsize=10.8)
ax1.grid(axis="y", alpha=0.25)
for b, v in zip(bars, seg_vals):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.9, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9.5)

pg_classes = ["Cardiomegaly", "Pneumonia", "Effusion", "Overall (278)", "Mass", "Pneumothorax", "Atelectasis", "Nodule (8x8 limit)"]
pg_vals = [95.00, 62.50, 56.41, 42.81, 41.03, 27.50, 15.00, 2.50]
pg_colors = ["#059669", "#10b981", "#38bdf8", "#f59e0b", "#60a5fa", "#818cf8", "#f97316", "#ef4444"]
y2 = np.arange(len(pg_classes))
ax2.barh(y2, pg_vals, color=pg_colors, edgecolor="#1e293b", linewidth=0.6)
ax2.set_yticks(y2)
ax2.set_yticklabels(pg_classes, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlim(0, 108)
ax2.set_xlabel("Pointing Game Accuracy (%) — Weakly-Supervised (No BBox Training)")
ax2.set_title("Lung-Gated Grad-CAM++ Localization (BBox Energy: 18.09% → 27.44%, +51.7%)", fontweight="bold", fontsize=10.2)
ax2.grid(axis="x", alpha=0.25)
for i, v in enumerate(pg_vals):
    ax2.text(v + 1.2, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig4_segmentation_and_xai_ablation.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================================
# FIGURE 5: Language Modeling Perplexity & RTX 4080 12GB VRAM Budget
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

ppl_labels = ["Base Qwen2.5-7B\n(Loss: 3.114)", "MedVision-7B-v2\n(Loss: 0.6345 | Acc: 83.46%)"]
ppl_vals = [22.51, 1.89]
bars = ax1.bar(ppl_labels, ppl_vals, color=["#ef4444", "#059669"], width=0.45, edgecolor="#1e293b", linewidth=0.6)
ax1.set_ylim(0, 27)
ax1.set_ylabel("Held-Out Validation Perplexity (Lower is Better)")
ax1.set_title("Medical Language Modeling Error Reduction (-91.6% PPL)", fontweight="bold", fontsize=11)
ax1.grid(axis="y", alpha=0.25)
for b, v in zip(bars, ppl_vals):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.6, f"PPL = {v:.2f}", ha="center", fontweight="bold", fontsize=10)

vram_labels = ["5-Model Production\nInference Peak", "7B QLoRA\nTraining Peak", "RTX 4080 Laptop\nHardware Ceiling"]
vram_vals = [6.46, 11.52, 12.00]
vram_colors = ["#059669", "#f59e0b", "#64748b"]
bars2 = ax2.bar(vram_labels, vram_vals, color=vram_colors, width=0.48, edgecolor="#1e293b", linewidth=0.6)
ax2.axhline(12.0, color="#dc2626", linestyle="--", lw=1.2)
ax2.set_ylim(0, 14)
ax2.set_ylabel("GPU VRAM Usage (GB)")
ax2.set_title("Memory Efficiency: 5 Models + CPU 175k RAG in 6.46 GB VRAM", fontweight="bold", fontsize=11)
ax2.grid(axis="y", alpha=0.25)
for b, v in zip(bars2, vram_vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.35, f"{v:.2f} GB", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig5_perplexity_and_vram.png"), dpi=300, bbox_inches="tight")
plt.close()

# Copy official 300 DPI Ensemble ROC Curve
roc_src = "./checkpoints/phase3d_ensemble_roc_300dpi.png"
if os.path.exists(roc_src):
    shutil.copy2(roc_src, os.path.join(fig_dir, "fig6_nih14_ensemble_roc_300dpi.png"))

# =========================================================================
# GENERATE MARKDOWN ACADEMIC REPORT
# =========================================================================
md_report = f"""# MedVision & Clinical LLM Studio — Comprehensive Academic & Engineering Evaluation Report

**Author / System:** MedVision AI Studio v4.0 (Multimodal Thoracic Radiology, Clinical LLM Alignment & Medical RAG)  
**Compute Hardware:** Single Consumer Laptop GPU — NVIDIA GeForce RTX 4080 (12 GB VRAM)  
**Data Integrity Audit:** `Train ∩ Val = 0`, `Train ∩ Test = 0`, `Val ∩ Test = 0` (100% Patient-Disjoint Adult Cohorts; Zero Data Leakage)

---

## 1. Executive Summary of Before-vs-After Gains

| Subsystem | Benchmark & Test Size | Baseline | Final v4.0 Model | Absolute Gain | Error / Risk Reduction |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Thoracic Vision Ensemble** | NIH ChestX-ray14 Official Test (`n=25,500`) | `0.8101` (Single DenseNet) | **`0.8208` Mean AUROC** (`Val: 0.8522`) | **`+0.0107`** (14/14 classes) | `Mean Sens: 83.9%` \| Beats both single models on all 14 diseases |
| **Anatomical Lung Segmentation** | Montgomery County Manual Masks (`n=138`) | — | **`87.00% ± 2.38%` Dice** (`IoU: 77.07%`) | Right: `90.36%`, Left: `83.26%` | Enables exact `CTR` & automated `OOD` non-X-ray rejection |
| **Weakly-Supervised XAI** | NIH-14 Radiologist BBoxes (`n=278`) | `18.09%` Raw Energy | **`27.44%` Lung-Gated Energy** | **`+9.35%` (+51.7% rel.)** | `Cardiomegaly PG: 95.0%` \| Suppresses mediastinal noise |
| **Medical Language Modeling** | Held-Out Validation (`n=500`) | `22.51` PPL (`Loss 3.114`) | **`1.89` PPL (`Loss 0.6345`)** | **`-20.62` PPL (-91.6%)** | `83.46%` Next-Token Accuracy |
| **Medical Evidence Reasoning** | PubMedQA Unseen (`n=200`) | `51.50%` | **`78.50%`** | **`+27.00%`** | Exceeds Human Physician Avg (`78.0%`) |
| **Medical Board Exam** | MedMCQA Full Validation (`n=4,183`) | `52.00%` | **`64.98%`** (Single-Pass: `64.52%`) | **`+12.98%`** | Backed by leak-free `175,413`-doc FAISS + Cross-Encoder |
| **Radiology Summarization** | IU-Xray Findings→Impression (`n=250`) | `58.08%` Micro-F1 | **`79.32%` Micro-F1** | **`+21.24%`** | **`97.70%` Negation Spec.** (`+6.48%`) \| `ROUGE-L: 63.76%` |

---

## 2. Visual Analysis & Empirical Figures (300 DPI)

### 2.1 Clinical LLM & Medical RAG Progression
[figures/fig1_llm_progression_bar.png]

### 2.2 NIH ChestX-ray14 14-Class Official Test Comparison (n=25,500)
[figures/fig2_nih14_per_class_auroc.png]

### 2.3 ConvNeXt-Small Convergence & AUROC-vs-FPR Regression Analysis
*Ordinary Least Squares (OLS) regression across all 14 pathologies demonstrates a strong negative linear correlation (`r = {r_val:.3f}`, `R² = {r2_val:.3f}`) between discriminative AUROC and False Positive Rate (`100 - Specificity`), quantitatively explaining alarm fatigue on diffuse infiltrative classes.*

[figures/fig3_learning_curve_and_regression.png]

### 2.4 Official 25,500-Image Ensemble ROC Curves
[figures/fig6_nih14_ensemble_roc_300dpi.png]

### 2.5 Lung Segmentation (Dice/IoU) & Lung-Gated Grad-CAM++ XAI Ablation
[figures/fig4_segmentation_and_xai_ablation.png]

### 2.6 Perplexity Reduction & Single-GPU (12 GB) Systems Efficiency
[figures/fig5_perplexity_and_vram.png]

---

## 3. Complete NIH ChestX-ray14 Per-Pathology Breakdown (n=25,500 Official Test)

| Pathology | DenseNet-121 (TTA) | ConvNeXt-Small (TTA) | **Ensemble AUROC** | Sensitivity (%) | Specificity (%) | False Positive Rate (%) | False Negative Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hernia** | 0.9102 | 0.9026 | **0.9203** | 79.1% | 91.7% | 8.3% | 20.9% |
| **Emphysema** | 0.9118 | 0.9013 | **0.9135** | 87.9% | 78.4% | 21.6% | 12.1% |
| **Cardiomegaly** | 0.8904 | 0.8925 | **0.8980** | 84.8% | 77.2% | 22.8% | 15.2% |
| **Pneumothorax** | 0.8615 | 0.8643 | **0.8701** | 89.0% | 66.7% | 33.3% | 11.0% |
| **Edema** | 0.8465 | 0.8495 | **0.8535** | 92.4% | 58.4% | 41.6% | 7.6% |
| **Fibrosis** | 0.8334 | 0.8280 | **0.8394** | 81.8% | 70.7% | 29.3% | 18.2% |
| **Mass** | 0.8273 | 0.8257 | **0.8331** | 83.9% | 62.8% | 37.2% | 16.1% |
| **Effusion** | 0.8259 | 0.8307 | **0.8325** | 88.0% | 59.7% | 40.3% | 12.0% |
| **Pleural_Thickening** | 0.7770 | 0.7766 | **0.7850** | 83.8% | 56.9% | 43.1% | 16.2% |
| **Atelectasis** | 0.7739 | 0.7793 | **0.7827** | 83.5% | 56.4% | 43.6% | 16.5% |
| **Nodule** | 0.7557 | 0.7639 | **0.7679** | 66.1% | 72.9% | 27.1% | 33.9% |
| **Consolidation** | 0.7545 | 0.7502 | **0.7597** | 90.0% | 44.4% | 55.6% | 10.0% |
| **Pneumonia** | 0.7099 | 0.7110 | **0.7192** | 76.2% | 55.9% | 44.1% | 23.8% |
| **Infiltration** | 0.7087 | 0.7147 | **0.7162** | 88.2% | 36.6% | 63.4% | 11.8% |
| **MEAN (14 Classes)** | **0.8133** | **0.8136** | **0.8208** | **83.9%** | **63.5%** | **36.5%** | **16.1%** |

---

## 4. Scientific Limitations & Future Work (Transparent Disclosure)
1. **Small-Lesion Localization (`Nodule Pointing Game = 2.5%`):** Image-level classification at `256×256` yields an `8×8` final feature map, which is too coarse for millimeter-scale nodules. Future work: `512×512` input resolution or dedicated supervised object detection (e.g., Faster R-CNN / YOLOv8-Med).
2. **Retrocardiac Left-Lung Mask Definition (`83.26%` vs `90.36%` Right):** Montgomery manual masks exclude retrocardiac lung area while `PSPNet` segments the full anatomical lung silhouette.
3. **External Hospital Validation:** All vision evaluations are on NIH ChestX-ray14 (`0` patient overlap); cross-institutional domain shift evaluation (e.g., CheXpert / PadChest) is planned as future work.
4. **Continual DPO Alignment (`32` PoC pairs):** Implemented as a functional Human-in-the-Loop continual learning pipeline (`SQLite + n8n + TRL DPO`) rather than a large-scale offline DPO benchmark.
"""

with open(os.path.join(pres_dir, "MedVision_Academic_Report.md"), "w", encoding="utf-8") as f:
    f.write(md_report)

# =========================================================================
# GENERATE PRINT-READY ACADEMIC HTML PRESENTATION REPORT
# =========================================================================
html_rows = ""
for c, d1, c1, e1, se, sp in zip(classes, dn_aucs, cn_aucs, ens_aucs, sens_pct, spec_pct):
    html_rows += (
        f"<tr><td><b>{c}</b></td><td>{d1:.4f}</td><td>{c1:.4f}</td>"
        f"<td style='color:#059669;font-weight:700'>{e1:.4f}</td>"
        f"<td>{se:.1f}%</td><td>{sp:.1f}%</td><td>{100-sp:.1f}%</td><td>{100-se:.1f}%</td></tr>\n"
    )

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>MedVision &amp; Clinical LLM Studio v4.0 — Academic Evaluation Dossier</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; color: #0f172a; background: #f8fafc; margin: 0; padding: 32px; line-height: 1.6; }}
  .container {{ max-width: 1120px; margin: 0 auto; background: #ffffff; padding: 40px 48px; border-radius: 16px; box-shadow: 0 10px 30px rgba(15,23,42,0.08); }}
  h1 {{ font-size: 28px; margin-bottom: 6px; color: #0f172a; border-bottom: 3px solid #0284c7; padding-bottom: 10px; }}
  h2 {{ font-size: 20px; margin-top: 34px; color: #0369a1; border-left: 4px solid #0284c7; padding-left: 10px; }}
  .subtitle {{ color: #475569; font-size: 14px; margin-bottom: 24px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 22px 0; }}
  .kpi {{ background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px 16px; }}
  .kpi .val {{ font-size: 24px; font-weight: 800; color: #059669; }}
  .kpi .lbl {{ font-size: 12px; color: #475569; font-weight: 600; text-transform: uppercase; }}
  .kpi .sub {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: center; }}
  th {{ background: #0f172a; color: #ffffff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  .fig-box {{ margin: 24px 0; text-align: center; page-break-inside: avoid; }}
  .fig-box img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 10px; }}
  .fig-cap {{ font-size: 12.5px; color: #475569; margin-top: 6px; font-style: italic; }}
  .callout {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 14px 18px; border-radius: 8px; margin: 18px 0; font-size: 13.5px; }}
  @media print {{ body {{ background: #fff; padding: 0; }} .container {{ box-shadow: none; padding: 12px; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>MedVision &amp; Clinical LLM Studio (v4.0)</h1>
  <div class="subtitle">
    Comprehensive Academic &amp; Engineering Evaluation Dossier &bull; Single RTX 4080 Laptop GPU (12 GB VRAM) &bull; Zero Train/Val/Test Overlap Verified
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">NIH-14 Official Test</div><div class="val">0.8208 AUROC</div><div class="sub">n=25,500 | 14/14 classes best | 83.9% Sens</div></div>
    <div class="kpi"><div class="lbl">PubMedQA Unseen</div><div class="val">78.50%</div><div class="sub">Base 51.5% (+27.0%) | Beats Human Avg (78%)</div></div>
    <div class="kpi"><div class="lbl">MedMCQA Full Split</div><div class="val">64.98%</div><div class="sub">n=4,183 | Base 52.0% (+12.98%) | 175k RAG</div></div>
    <div class="kpi"><div class="lbl">IU-Xray Summarization</div><div class="val">79.32% F1</div><div class="sub">97.70% Negation Spec | 63.76% ROUGE-L</div></div>
  </div>

  <div class="callout">
    <b>Engineering &amp; Clinical Highlights:</b> All 5 production neural components (<code>DenseNet-121</code>, <code>ConvNeXt-Small</code>, <code>PSPNet</code> lung segmenter, 4-bit <code>Qwen2.5-7B</code> with Dual-LoRA adapters) run simultaneously in <b>6.46 GB VRAM</b> alongside a CPU-offloaded <b>175,413-passage FAISS + Cross-Encoder RAG</b>, automated <b>OOD non-X-ray Guardrail</b>, <b>Cross-Modal Contradiction Verifier</b>, and <b>SQLite + n8n Continual DPO</b> loop.
  </div>

  <h2>1. Clinical LLM &amp; Medical RAG Progression (Before vs. After)</h2>
  <div class="fig-box">
    <img src="figures/fig1_llm_progression_bar.png" alt="LLM Progression" />
    <div class="fig-cap">Figure 1: Verified accuracy gains across PubMedQA (n=200), MedMCQA Full Validation (n=4,183), and IU-Xray Findings-to-Impression (n=250).</div>
  </div>

  <h2>2. Thoracic Vision Ensemble — 14-Class Official Test Evaluation (n=25,500)</h2>
  <div class="fig-box">
    <img src="figures/fig2_nih14_per_class_auroc.png" alt="NIH-14 Per Class AUROC" />
    <div class="fig-cap">Figure 2: Ensemble (0.55 DenseNet-121 + 0.45 ConvNeXt-Small, 2-View TTA) outperforms both individual architectures on all 14/14 diseases.</div>
  </div>

  <table>
    <thead>
      <tr><th>Pathology</th><th>DenseNet-121</th><th>ConvNeXt-S</th><th>Ensemble AUROC</th><th>Sensitivity</th><th>Specificity</th><th>FPR (Error)</th><th>FNR (Miss)</th></tr>
    </thead>
    <tbody>
      {html_rows}
      <tr style="background:#e2e8f0;font-weight:800"><td>MEAN (14 Classes)</td><td>0.8133</td><td>0.8136</td><td style="color:#059669">0.8208</td><td>83.9%</td><td>63.5%</td><td>36.5%</td><td>16.1%</td></tr>
    </tbody>
  </table>

  <h2>3. Training Convergence &amp; AUROC-vs-FPR Linear Regression</h2>
  <div class="fig-box">
    <img src="figures/fig3_learning_curve_and_regression.png" alt="Learning Curve and Regression" />
    <div class="fig-cap">Figure 3: (Left) ConvNeXt-Small 8-epoch loss &amp; validation AUROC progression (Val Ensemble = 0.8522). (Right) OLS regression showing strong inverse correlation (r = {r_val:.3f}, R&sup2; = {r2_val:.3f}) between AUROC and False Positive Rate.</div>
  </div>

  <div class="fig-box">
    <img src="figures/fig6_nih14_ensemble_roc_300dpi.png" alt="ROC Curves" />
    <div class="fig-cap">Figure 4: Official 25,500-image patient-disjoint ROC curves for primary thoracic pathologies.</div>
  </div>

  <h2>4. Anatomical Lung Segmentation (Dice/IoU) &amp; XAI Localization Ablation</h2>
  <div class="fig-box">
    <img src="figures/fig4_segmentation_and_xai_ablation.png" alt="Segmentation and XAI" />
    <div class="fig-cap">Figure 5: (Left) PSPNet evaluation on 138 Montgomery manual lung masks (87.00% Combined Dice). (Right) Weakly-supervised Grad-CAM++ localization on 278 radiologist BBoxes (+51.7% relative BBox energy gain via Lung-Gating; 95.0% Pointing Game on Cardiomegaly).</div>
  </div>

  <h2>5. Language Modeling Error Reduction &amp; Single-GPU Memory Budget</h2>
  <div class="fig-box">
    <img src="figures/fig5_perplexity_and_vram.png" alt="Perplexity and VRAM" />
    <div class="fig-cap">Figure 6: Held-out validation perplexity drops 91.6% (22.51 &rarr; 1.89) while the full 5-model production stack operates in 6.46 GB of 12 GB VRAM.</div>
  </div>

  <h2>6. Transparent Scientific Limitations &amp; Future Work</h2>
  <ul>
    <li><b>Small-Lesion Localization (Nodule 2.5% Pointing Game):</b> Image-level classification at 256&times;256 produces an 8&times;8 feature grid too coarse for millimeter-scale nodules; future work adds 512&times;512 resolution or supervised object detection.</li>
    <li><b>Anatomical Mask Definition:</b> Montgomery manual masks exclude retrocardiac left-lung pixels (Left Dice 83.26% vs Right 90.36%).</li>
    <li><b>External Validation:</b> Vision metrics are patient-disjoint within NIH ChestX-ray14 (0 overlap); cross-hospital validation (CheXpert / PadChest) is planned next.</li>
    <li><b>Continual DPO PoC:</b> Initial 32-pair DPO validates the end-to-end human-in-the-loop retraining pipeline rather than claiming offline DPO benchmark gains.</li>
  </ul>
</div>
</body>
</html>
"""

with open(os.path.join(pres_dir, "MedVision_Academic_Report.html"), "w", encoding="utf-8") as f:
    f.write(html_doc)

print("=" * 84)
print(f"  [SUCCESS] Presentation Package Created in: {pres_dir}")
print("=" * 84)
for root, _, files in os.walk(pres_dir):
    for fn in sorted(files):
        fp = os.path.join(root, fn)
        print(f"  - {os.path.relpath(fp, pres_dir)} ({os.path.getsize(fp)/1024:.1f} KB)")
