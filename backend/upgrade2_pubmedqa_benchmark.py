import json, os, torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def get_choice_logits(model, tokenizer, prompt, choice_ids):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1, :]
    return {k: float(logits[v].item()) for k, v in choice_ids.items()}

def evaluate_calibrated(model, tokenizer, samples, label):
    choice_ids = {
        "yes": tokenizer.encode("Yes", add_special_tokens=False)[0],
        "no": tokenizer.encode("No", add_special_tokens=False)[0],
        "maybe": tokenizer.encode("Maybe", add_special_tokens=False)[0],
    }

    # Content-Free Contextual Calibration (Zhao et al., ICML) to remove token prior bias
    cf_messages = [
        {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
        {"role": "user", "content": "Abstract:\nN/A\n\nQuestion: N/A\nAnswer (Yes, No, or Maybe):"}
    ]
    cf_prompt = tokenizer.apply_chat_template(cf_messages, tokenize=False, add_generation_prompt=True)
    prior_bias = get_choice_logits(model, tokenizer, cf_prompt, choice_ids)

    correct, total = 0, len(samples)
    pred_counts = Counter()
    gold_counts = Counter()

    for idx, ex in enumerate(samples):
        ctx = " ".join(ex["context"]["contexts"])[-2000:]
        q = ex["question"]
        gold = ex["final_decision"].strip().lower()
        gold_counts[gold] += 1

        messages = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {q}\nAnswer (Yes, No, or Maybe):"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_scores = get_choice_logits(model, tokenizer, prompt, choice_ids)
        cal_scores = {k: raw_scores[k] - prior_bias[k] for k in choice_ids}

        pred = max(cal_scores, key=cal_scores.get)
        pred_counts[pred] += 1
        if pred == gold:
            correct += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{label}] {idx+1}/{total} | Calibrated Accuracy: {correct/(idx+1)*100:.1f}% | Preds: {dict(pred_counts)}")
    return {"accuracy_pct": round(correct/total*100, 2), "correct": correct, "total": total, "pred_dist": dict(pred_counts), "gold_dist": dict(gold_counts)}

def main():
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    adapter_path = "./checkpoints/medvision-7b-stage2-dpo-final"
    n = 200
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(n))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base_path)

    print(f">>> [1/2] Evaluating Base Qwen2.5-7B-Instruct (Contextual Calibration, n={n})...")
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()
    r1 = evaluate_calibrated(base, tok, ds, "Base Qwen2.5-7B")

    print(f"\n>>> [2/2] Evaluating YOUR MedVision-7B-SFT-DPO (Contextual Calibration, n={n})...")
    aligned = PeftModel.from_pretrained(base, adapter_path).eval()
    r2 = evaluate_calibrated(aligned, tok, ds, "MedVision-7B-SFT-DPO")

    delta = round(r2["accuracy_pct"] - r1["accuracy_pct"], 2)
    out = {
        "benchmark": f"qiaojin/PubMedQA (pqa_labeled, n={n}, Contextual Calibration)",
        "base_accuracy_pct": r1["accuracy_pct"],
        "medvision_7b_sft_dpo_accuracy_pct": r2["accuracy_pct"],
        "delta_pct": delta,
        "gold_distribution": r1["gold_dist"],
        "base_pred_distribution": r1["pred_dist"],
        "aligned_pred_distribution": r2["pred_dist"],
    }
    with open("./checkpoints/upgrade2_pubmedqa_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "="*80)
    print(f"  Gold Label Distribution                   : {r1['gold_dist']}")
    print(f"  Base Qwen2.5-7B-Instruct (Calibrated)     : {r1['accuracy_pct']}% ({r1['correct']}/{n})")
    print(f"  MedVision-7B-SFT-DPO (Calibrated, Yours)  : {r2['accuracy_pct']}% ({r2['correct']}/{n})  [Delta: {delta:+.2f}%]")
    print("="*80)

if __name__ == "__main__":
    main()
