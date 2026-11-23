import gc, json, os, re, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("=" * 90)
print("  REVIEWER #2 VALIDATION SUITE: 95% CI / p-values + Fair PubMedQA + 100-Case E2E Multimodal")
print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 90)

# =========================================================================
# PART 1: NIH-14 Official Test (n=25,500) — 95% Bootstrap CI & Paired p-value
# =========================================================================
print("\n[1/3] Computing Paired Bootstrap 95% CI & p-values on NIH-14 Official Test (n=25,500)...")
with open("./nih_cxr14_cache/splits_meta.json", "r", encoding="utf-8") as f:
    nih_meta = json.load(f)
with open("./checkpoints/medvision_nih14_ensemble_meta.json", "r", encoding="utf-8") as f:
    ens_meta = json.load(f)

classes = nih_meta.get("classes", [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
])
img_dir = "./nih_cxr14_cache/images"

class NIH14DS(Dataset):
    def __init__(self, recs, tf):
        self.recs = recs; self.tf = tf
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return self.tf(Image.open(os.path.join(img_dir, r["file"])).convert("RGB")), torch.tensor(r["labels"], dtype=torch.float32)

tf224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@torch.no_grad()
def infer_tta(model, loader):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = 0.5 * (torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.vstack(ps), np.vstack(ys)

dn_ckpt = torch.load("./checkpoints/medvision_nih14_densenet121.pth", map_location=device, weights_only=False)
dn_state = dn_ckpt["model_state_dict"] if "model_state_dict" in dn_ckpt else dn_ckpt
dn = tv_models.densenet121(weights=None)
in_dn = dn.classifier.in_features
dn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dn, len(classes))) if "classifier.1.weight" in dn_state else nn.Linear(in_dn, len(classes))
dn.load_state_dict(dn_state); dn = dn.to(device).eval()

te224 = DataLoader(NIH14DS(nih_meta["test"], tf224), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
p_dn, y_te = infer_tta(dn, te224)
del dn; gc.collect(); torch.cuda.empty_cache()

cn_ckpt = torch.load("./checkpoints/medvision_nih14_convnext_small.pth", map_location=device, weights_only=False)
cn = tv_models.convnext_small(weights=None)
cn.classifier[2] = nn.Sequential(nn.Dropout(0.2), nn.Linear(cn.classifier[2].in_features, len(classes)))
cn.load_state_dict(cn_ckpt["model_state_dict"]); cn = cn.to(device).eval()

te256 = DataLoader(NIH14DS(nih_meta["test"], tf256), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
p_cn, _ = infer_tta(cn, te256)
del cn; gc.collect(); torch.cuda.empty_cache()

w_cn = float(ens_meta.get("convnext_weight", 0.45))
w_dn = float(ens_meta.get("densenet_weight", 0.55))
p_ens = w_cn * p_cn + w_dn * p_dn

def mean_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1])]))

rng = np.random.default_rng(42)
n_boot = 300
N = len(y_te)
boot_dn, boot_cn, boot_ens = [], [], []
for _ in range(n_boot):
    idx = rng.integers(0, N, size=N)
    yb = y_te[idx]
    if any(yb[:, c].sum() == 0 or yb[:, c].sum() == N for c in range(yb.shape[1])):
        continue
    boot_dn.append(mean_auc(yb, p_dn[idx]))
    boot_cn.append(mean_auc(yb, p_cn[idx]))
    boot_ens.append(mean_auc(yb, p_ens[idx]))

boot_dn, boot_cn, boot_ens = np.array(boot_dn), np.array(boot_cn), np.array(boot_ens)
diff_vs_dn = boot_ens - boot_dn
diff_vs_cn = boot_ens - boot_cn
p_vs_dn = float(2.0 * min((diff_vs_dn <= 0).mean(), (diff_vs_dn >= 0).mean()))
p_vs_cn = float(2.0 * min((diff_vs_cn <= 0).mean(), (diff_vs_cn >= 0).mean()))

nih_stats = {
    "n_test": int(N),
    "n_bootstrap": int(len(boot_ens)),
    "densenet_tta_mean": round(float(mean_auc(y_te, p_dn)), 4),
    "densenet_tta_ci95": [round(float(np.percentile(boot_dn, 2.5)), 4), round(float(np.percentile(boot_dn, 97.5)), 4)],
    "convnext_tta_mean": round(float(mean_auc(y_te, p_cn)), 4),
    "convnext_tta_ci95": [round(float(np.percentile(boot_cn, 2.5)), 4), round(float(np.percentile(boot_cn, 97.5)), 4)],
    "ensemble_mean": round(float(mean_auc(y_te, p_ens)), 4),
    "ensemble_ci95": [round(float(np.percentile(boot_ens, 2.5)), 4), round(float(np.percentile(boot_ens, 97.5)), 4)],
    "delta_vs_densenet_tta": round(float(diff_vs_dn.mean()), 4),
    "p_value_vs_densenet_tta": max(p_vs_dn, 1.0 / (len(boot_ens) + 1)),
    "delta_vs_convnext_tta": round(float(diff_vs_cn.mean()), 4),
    "p_value_vs_convnext_tta": max(p_vs_cn, 1.0 / (len(boot_ens) + 1)),
}
print("  [NIH-14 STATS]", json.dumps(nih_stats, indent=2))

# =========================================================================
# PART 2: Fair Logit-Scored PubMedQA Baseline vs MedVision-7B-v2 (n=200)
# =========================================================================
print("\n[2/3] Fair Logit-Scored PubMedQA Evaluation (Base Qwen2.5-7B vs MedVision-7B-v2, n=200 Unseen)...")
ds_pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
pqa_test = [ds_pqa[i] for i in range(200)]

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("./qwen7b_local")
if tok.pad_token is None: tok.pad_token = tok.eos_token
base_m = AutoModelForCausalLM.from_pretrained("./qwen7b_local", quantization_config=bnb_cfg, device_map="auto", dtype=torch.bfloat16)
peft_m = PeftModel.from_pretrained(base_m, "./checkpoints/medvision-7b-v2-multitask-sft")
peft_m.eval()

cand_words = ["yes", "no", "maybe"]
cand_ids = [tok.encode(w, add_special_tokens=False)[0] for w in cand_words]

def eval_pubmedqa_fair(model_obj, disable_adapter=False):
    preds, golds = [], []
    with torch.inference_mode():
        for ex in pqa_test:
            ctx = " ".join(ex["context"]["contexts"])[:2800]
            user_msg = (
                f"Abstract Context:\n{ctx}\n\n"
                f"Question: {ex['question']}\n"
                f"Based strictly on the abstract above, answer with a single word: yes, no, or maybe."
            )
            txt = tok.apply_chat_template(
                [{"role": "system", "content": "You are an expert biomedical researcher. Reply with only one word: yes, no, or maybe."},
                 {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True
            )
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
            if disable_adapter:
                with model_obj.disable_adapter():
                    logits = model_obj(**enc, use_cache=False).logits[0, -1, cand_ids]
            else:
                logits = model_obj(**enc, use_cache=False).logits[0, -1, cand_ids]
            pred = cand_words[int(torch.argmax(logits).item())]
            preds.append(pred)
            golds.append(ex["final_decision"].lower())
    corr = np.array([p == g for p, g in zip(preds, golds)], dtype=np.float64)
    return corr

corr_base = eval_pubmedqa_fair(peft_m, disable_adapter=True)
corr_v2   = eval_pubmedqa_fair(peft_m, disable_adapter=False)

# Bootstrap 95% CI & McNemar exact-like chi2 p-value
b_base = [corr_base[rng.integers(0, 200, 200)].mean() * 100 for _ in range(1000)]
b_v2   = [corr_v2[rng.integers(0, 200, 200)].mean() * 100 for _ in range(1000)]
n01 = int(((corr_base == 0) & (corr_v2 == 1)).sum())
n10 = int(((corr_base == 1) & (corr_v2 == 0)).sum())
mcnemar_chi2 = ((abs(n01 - n10) - 1) ** 2) / max(n01 + n10, 1)
# Normal approximation two-sided p-value from chi2 (1 df)
import math
p_mcnemar = float(math.erfc(math.sqrt(mcnemar_chi2 / 2.0)))

pqa_stats = {
    "n_unseen_test": 200,
    "fair_base_qwen7b_acc_pct": round(float(corr_base.mean() * 100), 2),
    "fair_base_ci95_pct": [round(float(np.percentile(b_base, 2.5)), 2), round(float(np.percentile(b_base, 97.5)), 2)],
    "medvision_v2_acc_pct": round(float(corr_v2.mean() * 100), 2),
    "medvision_v2_ci95_pct": [round(float(np.percentile(b_v2, 2.5)), 2), round(float(np.percentile(b_v2, 97.5)), 2)],
    "absolute_gain_pct": round(float((corr_v2.mean() - corr_base.mean()) * 100), 2),
    "mcnemar_n01_improved": n01,
    "mcnemar_n10_regressed": n10,
    "mcnemar_p_value": round(p_mcnemar, 6),
}
print("  [PUBMEDQA FAIR STATS]", json.dumps(pqa_stats, indent=2))

del peft_m, base_m; gc.collect(); torch.cuda.empty_cache()

# =========================================================================
# PART 3: 100-Case End-to-End Multimodal Validation (Raw Image -> Vision+CTR -> LLM)
# =========================================================================
print("\n[3/3] Running 100-Case End-to-End Multimodal Pipeline Validation on Unseen IU-Xray Images...")
from app.services.vision_ensemble import analyze_chest_xray_ensemble
from app.services.clinical_llm_rag import synthesize_radiology_impression

ds_iu = load_dataset("ykumards/open-i", split="train")
# Take last 100 valid frontal X-rays (strictly outside the first 3,080 train split)
e2e_samples = []
for i in range(len(ds_iu)-1, -1, -1):
    ex = ds_iu[i]
    img = ex.get("img_frontal")
    imp = (ex.get("impression") or "").strip()
    if img is not None and len(imp) > 10:
        e2e_samples.append((img.convert("RGB"), imp))
    if len(e2e_samples) >= 100:
        break

latencies_ms = []
contradictions = 0
normal_gt, normal_pred_ok = 0, 0
abnormal_gt, abnormal_detected = 0, 0
ctr_enlarged_mentions = 0
ctr_enlarged_total = 0

for idx, (pil_img, gt_imp) in enumerate(e2e_samples, 1):
    t0 = time.perf_counter()
    vres = analyze_chest_xray_ensemble(pil_img, operating_mode="balanced")
    if vres.get("ood_rejected"):
        continue
    top_findings = [(f["name"], f["prob"]) for f in vres["findings"] if f["prob"] >= 0.45] or vres["top3"]
    ctr_str = str(vres["ctr"]) if vres["ctr"] is not None else "N/A"
    lres = synthesize_radiology_impression(
        primary=vres["primary_finding"],
        top_conf=vres["top_confidence"],
        top3=top_findings,
        urgency=vres["urgency"],
        patient_ctx=f"IU-Xray Test Case #{idx}, Exact PSPNet CTR={ctr_str}",
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    latencies_ms.append(dt_ms)

    rep_low = lres["report"].lower()
    # Check Cross-Modal Contradiction (did LLM negate any vision finding >= 60%?)
    pos_vis = [n for n, p in top_findings if p >= 0.60]
    if any(f"no {n.lower().replace('_', ' ')}" in rep_low for n in pos_vis):
        contradictions += 1

    # Check CTR > 0.50 fusion propagation
    if vres["ctr"] is not None and vres["ctr"] > 0.50:
        ctr_enlarged_total += 1
        if "cardiomegaly" in rep_low or "enlarged" in rep_low or "ctr" in rep_low:
            ctr_enlarged_mentions += 1

    # Check Normal vs Abnormal clinical triage agreement vs Ground Truth Impression
    gt_low = gt_imp.lower()
    is_gt_normal = any(k in gt_low for k in ("no acute", "normal", "unremarkable", "clear lungs")) and not any(
        k in gt_low for k in ("pneumothorax", "effusion", "edema", "pneumonia", "cardiomegaly", "mass", "consolidation")
    )
    if is_gt_normal:
        normal_gt += 1
        if vres["urgency"] == "ROUTINE" or "no acute" in rep_low or "normal" in rep_low:
            normal_pred_ok += 1
    else:
        abnormal_gt += 1
        if vres["urgency"] in ("HIGH", "STAT") or len(pos_vis) > 0:
            abnormal_detected += 1

    if idx % 25 == 0:
        print(f"  [E2E {idx}/100] Mean Latency: {np.mean(latencies_ms):.0f}ms | Contradictions: {contradictions}")

e2e_stats = {
    "n_evaluated": len(latencies_ms),
    "mean_e2e_latency_ms": round(float(np.mean(latencies_ms)), 1),
    "cross_modal_contradiction_rate_pct": round(contradictions / max(len(latencies_ms), 1) * 100, 2),
    "ctr_gt_050_total": ctr_enlarged_total,
    "ctr_gt_050_reported_by_llm_pct": round(ctr_enlarged_mentions / max(ctr_enlarged_total, 1) * 100, 1),
    "normal_gt_cases": normal_gt,
    "normal_triage_agreement_pct": round(normal_pred_ok / max(normal_gt, 1) * 100, 1),
    "abnormal_gt_cases": abnormal_gt,
    "abnormal_triage_sensitivity_pct": round(abnormal_detected / max(abnormal_gt, 1) * 100, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
}
print("  [E2E MULTIMODAL STATS]", json.dumps(e2e_stats, indent=2))

final_out = {"nih14_stats": nih_stats, "pubmedqa_fair_stats": pqa_stats, "e2e_multimodal_stats": e2e_stats}
with open("./checkpoints/reviewer2_stats_e2e.json", "w", encoding="utf-8") as f:
    json.dump(final_out, f, indent=2)
print("\n[SAVED] All Reviewer #2 statistics -> ./checkpoints/reviewer2_stats_e2e.json")
