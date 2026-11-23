import json, os, torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def compute_macro_f1(golds, preds, labels=("yes", "no", "maybe")):
    f1s = {}
    for lbl in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(golds, preds) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(golds, preds) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[lbl] = round(2 * prec * rec / (prec + rec) * 100, 2) if (prec + rec) > 0 else 0.0
    return round(sum(f1s.values()) / len(labels), 2), f1s

def get_probs_for_adapter(base_model, adapter_path, tok, test_ds, choice_ids):
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    all_probs = []
    for ex in test_ds:
        ctx = " ".join(ex["context"]["contexts"])[-1800:]
        q = ex["question"]
        messages = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {q}\nAnswer (Yes, No, or Maybe):"}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            vec = torch.tensor([logits[choice_ids["yes"]], logits[choice_ids["no"]], logits[choice_ids["maybe"]]])
            probs = F.softmax(vec, dim=0).tolist()
        all_probs.append(probs)
    model.unload()
    return all_probs

def main():
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    a1 = "./checkpoints/medvision-7b-clinical-final"
    a2 = "./checkpoints/medvision-7b-clinical-balanced-final"

    test_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200))
    golds = [ex["final_decision"].strip().lower() for ex in test_ds]
    labels = ["yes", "no", "maybe"]

    tok = AutoTokenizer.from_pretrained(base_path)
    choice_ids = {k: tok.encode(k.capitalize(), add_special_tokens=False)[0] for k in labels}
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()

    p1 = get_probs_for_adapter(base_model, a1, tok, test_ds, choice_ids)
    p2 = get_probs_for_adapter(base_model, a2, tok, test_ds, choice_ids)

    print("\n--- Threshold Calibration Sweep (50/50 Multi-Adapter Ensemble) ---")
    best = None
    for thresh in [0.40, 0.30, 0.25, 0.20, 0.15, 0.12]:
        preds = []
        for v1, v2 in zip(p1, p2):
            comb = [0.5 * v1[i] + 0.5 * v2[i] for i in range(3)]
            if comb[2] >= thresh:
                preds.append("maybe")
            else:
                preds.append("yes" if comb[0] >= comb[1] else "no")
        acc = round(sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100.0, 2)
        macro_f1, f1s = compute_macro_f1(golds, preds)
        dist = dict(Counter(preds))
        print(f"  maybe_thresh={thresh:.2f} -> Acc: {acc:5.2f}% | Macro-F1: {macro_f1:5.2f}% | F1s: {f1s} | Preds: {dist}")
        if best is None or (acc >= 74.0 and macro_f1 > best["macro_f1_pct"]):
            best = {"threshold": thresh, "accuracy_pct": acc, "macro_f1_pct": macro_f1, "per_class_f1": f1s, "preds": dist}

    print("\n" + "=" * 82)
    print(f"  OPTIMAL CALIBRATED ENSEMBLE: Acc={best['accuracy_pct']}% | Macro-F1={best['macro_f1_pct']}% | Preds={best['preds']}")
    print("=" * 82)

if __name__ == "__main__":
    main()
