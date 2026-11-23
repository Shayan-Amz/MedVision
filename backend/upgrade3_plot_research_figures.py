import json, os, matplotlib.pyplot as plt

with open("./checkpoints/upgrade1_two_stage_report.json") as f:
    r1 = json.load(f)

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), dpi=300)
fig.patch.set_facecolor("#070b14")

# Panel 1: Real Train vs Held-Out Val Loss Curve (5000 samples, r=32)
ax = axes[0]
ax.set_facecolor("#0f172a")
t_steps = [x["step"] for x in r1["stage1_sft"]["train_logs"]]
t_loss  = [x["loss"] for x in r1["stage1_sft"]["train_logs"]]
e_steps = [x["step"] for x in r1["stage1_sft"]["eval_logs"]]
e_loss  = [x["eval_loss"] for x in r1["stage1_sft"]["eval_logs"]]
ax.plot(t_steps, t_loss, color="#38bdf8", linewidth=2.0, label="Train Loss (5k Medical)")
ax.plot(e_steps, e_loss, color="#34d399", marker="o", linewidth=2.2, label="Held-Out Val Loss (n=300)")
ax.set_title("1. Qwen2.5-7B 4-bit QLoRA Convergence", fontsize=11, color="white", pad=10)
ax.set_xlabel("Optimization Step (RTX 4080 BF16)", fontsize=9, color="#94a3b8")
ax.set_ylabel("Cross-Entropy Loss", fontsize=9, color="#94a3b8")
ax.grid(alpha=0.18)
ax.legend(fontsize=8.5, frameon=False)

# Panel 2: Ablation Study — Held-Out Perplexity Drop
ax = axes[1]
ax.set_facecolor("#0f172a")
labels = ["Base\nQwen2.5-7B", "SFT 1k\n(r=16)", "SFT 5k + DPO\n(r=32)"]
ppls = [22.51, 1.98, r1["stage1_sft"]["val_perplexity"]]
colors = ["#64748b", "#38bdf8", "#10b981"]
bars = ax.bar(labels, ppls, color=colors, width=0.50)
for b, v in zip(bars, ppls):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}", ha="center", color="white", fontweight="bold", fontsize=10)
ax.set_title("2. Ablation: Held-Out Perplexity (Lower=Better)", fontsize=11, color="white", pad=10)
ax.set_ylabel("Validation Perplexity", fontsize=9, color="#94a3b8")
ax.set_ylim(0, 26)
ax.grid(axis="y", alpha=0.18)

# Panel 3: PubMedQA Official Clinical Benchmark (Accuracy & Calibrated Ensemble vs Human Expert)
ax = axes[2]
ax.set_facecolor("#0f172a")
b_labels = ["Base 7B\n(Calibrated)", "Ensemble 7B\n(F1=59.1%)", "MedVision-7B\n(Final SFT+DPO)"]
accs = [51.5, 71.5, 77.0]
bars2 = ax.bar(b_labels, accs, color=["#64748b", "#38bdf8", "#10b981"], width=0.48)
ax.axhline(78.0, color="#f59e0b", linestyle="--", linewidth=1.6, label="Human Physician Avg (78.0%)")
for b, v in zip(bars2, accs):
    ax.text(b.get_x() + b.get_width()/2, v + 1.8, f"{v:.1f}%", ha="center", color="white", fontweight="bold", fontsize=10)
ax.set_title("3. PubMedQA Benchmark (n=200 Unseen, +25.5%)", fontsize=11, color="white", pad=10)
ax.set_ylabel("Clinical Reasoning Accuracy (%)", fontsize=9, color="#94a3b8")
ax.set_ylim(0, 92)
ax.grid(axis="y", alpha=0.18)
ax.legend(fontsize=8.5, loc="lower right", frameon=False)

plt.tight_layout()
out_img = "./checkpoints/medvision_research_results_300dpi.png"
plt.savefig(out_img, dpi=300, facecolor="#070b14")
print("Saved publication figure:", os.path.abspath(out_img))
os.startfile(os.path.abspath(out_img))
