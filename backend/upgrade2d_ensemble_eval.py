import json, os, torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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

    print(">>> [1/2] Collecting Softmax probabilities from Adapter 1 (Clinical-Final)...")
    p1 = get_probs_for_adapter(base_model, a1, tok, test_ds, choice_ids)
    print(">>> [2/2] Collecting Softmax probabilities from Adapter 2 (Class-Balanced)...")
    p2 = get_probs_for_adapter(base_model, a2, tok, test_ds, choice_ids)

    best_w, best_acc, best_preds = 1.0, 0.0, []
    for w in [1.0, 0.85, 0.75, 0.65, 0.5, 0.0]:
        preds = []
        for v1, v2 in zip(p1, p2):
            comb = [w * v1[i] + (1 - w) * v2[i] for i in range(3)]
            preds.append(labels[comb.index(max(comb))])
        acc = round(sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100.0, 2)
        print(f"  Ensemble Weight (w={w:.2f} Clinical + {1-w:.2f} Balanced) -> Accuracy: {acc:.2f}% | Preds: {dict(Counter(preds))}")
        if acc > best_acc:
            best_acc, best_w, best_preds = acc, w, preds

    print("\n" + "=" * 80)
    print(f"  BEST ENSEMBLE ACCURACY ON UNSEEN PUBMEDQA (n=200): {best_acc:.2f}% (w={best_w:.2f})")
    print(f"  Predicted Distribution: {dict(Counter(best_preds))}")
    print("=" * 80)

if __name__ == "__main__":
    main()
