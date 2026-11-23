import json, math, os
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import f1_score
from scipy.stats import binomtest

BASE_DIR = "./qwen7b_local"
ADAPTER_DIR = "./checkpoints/medvision-7b-v2-multitask-sft"

ds_shuf = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
test_set = ds_shuf.select(range(0, 200))
train_set = ds_shuf.select(range(200, 1000))

test_pubids = [str(r["pubid"]) for r in test_set]
train_pubids = [str(r["pubid"]) for r in train_set]
assert len(set(test_pubids) & set(train_pubids)) == 0, "PubMedQA train/test leakage detected!"
print(f"PubMedQA Zero-Overlap Verified: Train={len(train_pubids)}, Test={len(test_pubids)}, Overlap=0")

def wilson_ci(k, n, z=1.959964):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = (z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)) / d
    return [round(100*(c-m), 2), round(100*(c+m), 2)]

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base_m = AutoModelForCausalLM.from_pretrained(BASE_DIR, quantization_config=bnb_cfg, device_map="auto", dtype=torch.bfloat16)
peft_m = PeftModel.from_pretrained(base_m, ADAPTER_DIR).eval()

cand_words = ["yes", "no", "maybe"]
cand_ids = [tok.encode(w, add_special_tokens=False)[0] for w in cand_words]

def run_fair(disable_adapter=False):
    preds, logits_all, golds = [], [], []
    with torch.inference_mode():
        for ex in test_set:
            ctx = " ".join(ex["context"]["contexts"])[:2800]
            user_msg = (
                f"Abstract Context:\n{ctx}\n\n"
                f"Question: {ex['question']}\n"
                f"Based strictly on the abstract above, answer with a single word: yes, no, or maybe."
            )
            txt = tok.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True)
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
            if disable_adapter:
                with peft_m.disable_adapter():
                    lg = peft_m(**enc, use_cache=False).logits[0, -1, cand_ids]
            else:
                lg = peft_m(**enc, use_cache=False).logits[0, -1, cand_ids]
            sc = {w: round(float(lg[i].item()), 4) for i, w in enumerate(cand_words)}
            preds.append(cand_words[int(torch.argmax(lg).item())])
            logits_all.append(sc)
            golds.append(ex["final_decision"].lower().strip())
    ok = sum(p == g for p, g in zip(preds, golds))
    acc = round(100.0 * ok / len(golds), 2)
    mf1 = round(100.0 * float(f1_score(golds, preds, labels=cand_words, average="macro", zero_division=0)), 2)
    pf1 = {w: round(100.0 * float(v), 2) for w, v in zip(cand_words, f1_score(golds, preds, labels=cand_words, average=None, zero_division=0))}
    return preds, logits_all, golds, ok, acc, mf1, pf1, dict(Counter(preds))

b_preds, b_lg, golds, b_ok, b_acc, b_mf1, b_pf1, b_dist = run_fair(disable_adapter=True)
f_preds, f_lg, _,     f_ok, f_acc, f_mf1, f_pf1, f_dist = run_fair(disable_adapter=False)

b01 = sum((bp != g and fp == g) for bp, fp, g in zip(b_preds, f_preds, golds))
b10 = sum((bp == g and fp != g) for bp, fp, g in zip(b_preds, f_preds, golds))
mc_p = float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if (b01 + b10) > 0 else 1.0

items = [
    {"idx": i, "pubid": test_pubids[i], "gold": golds[i],
     "base_pred": b_preds[i], "base_logits": b_lg[i],
     "ft_pred": f_preds[i], "ft_logits": f_lg[i]}
    for i in range(len(test_set))
]

out = {
    "dataset": "qiaojin/PubMedQA (pqa_labeled)",
    "split_protocol": "shuffle(seed=42).select(range(0, 200)) strictly disjoint from phase2a train range(200, 1000)",
    "train_test_pubid_overlap": 0,
    "n_test": 200,
    "rag_used": False,
    "base_model": BASE_DIR,
    "ft_checkpoint": ADAPTER_DIR,
    "base_acc_pct": b_acc, "base_ci95_wilson": wilson_ci(b_ok, 200), "base_macro_f1_pct": b_mf1, "base_per_class_f1": b_pf1, "base_pred_dist": b_dist,
    "ft_acc_pct": f_acc, "ft_ci95_wilson": wilson_ci(f_ok, 200), "ft_macro_f1_pct": f_mf1, "ft_per_class_f1": f_pf1, "ft_pred_dist": f_dist,
    "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_exact_p": round(mc_p, 8),
    "items": items,
}
with open("./checkpoints/pubmedqa_audit.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"Base Qwen2.5-7B  -> Acc: {b_acc}% ({b_ok}/200, 95% CI {out['base_ci95_wilson']}) | Macro-F1: {b_mf1}% | Dist: {b_dist}")
print(f"MedVision-7B-v2  -> Acc: {f_acc}% ({f_ok}/200, 95% CI {out['ft_ci95_wilson']}) | Macro-F1: {f_mf1}% | Per-F1: {f_pf1} | Dist: {f_dist}")
print(f"McNemar Exact    -> b01={b01}, b10={b10}, p={mc_p:.6g} | Saved checkpoints/pubmedqa_audit.json")
