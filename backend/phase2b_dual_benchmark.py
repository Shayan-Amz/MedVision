import json, os, torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    v2_adapter = "./checkpoints/medvision-7b-v2-multitask-sft"

    tok = AutoTokenizer.from_pretrained(base_path)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()
    model = PeftModel.from_pretrained(base, v2_adapter).eval()

    # 1. PubMedQA (Exact same 200 UNSEEN Test Questions)
    print("\n>>> [1/2] Evaluating MedVision-7B-v2 (12k Multi-Task) on 200 UNSEEN PubMedQA...")
    pqa_test = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200))
    c_ids = {k: tok.encode(k.capitalize(), add_special_tokens=False)[0] for k in ["yes", "no", "maybe"]}
    pqa_ok, pqa_preds = 0, []
    for ex in pqa_test:
        ctx = " ".join(ex["context"]["contexts"])[-1600:]
        msgs = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer (Yes, No, or Maybe):"}
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1, :]
            scores = {k: float(lg[v].item()) for k, v in c_ids.items()}
        pred = max(scores, key=scores.get)
        pqa_preds.append(pred)
        if pred == ex["final_decision"].strip().lower():
            pqa_ok += 1
    pqa_acc = round(pqa_ok / 200 * 100.0, 2)
    print(f"  [PubMedQA n=200 UNSEEN] Accuracy: {pqa_acc}% ({pqa_ok}/200) | Preds: {dict(Counter(pqa_preds))}")

    # 2. MedMCQA (200 Real 4-Choice Medical Entrance Exam Questions — Zero-Shot Generalization!)
    print("\n>>> [2/2] Evaluating MedVision-7B-v2 on 200 Real 4-Choice MedMCQA Exam Questions...")
    mcqa = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(200))
    opt_letters = ["A", "B", "C", "D"]
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in opt_letters]
    mcqa_ok = 0
    for idx, ex in enumerate(mcqa):
        q = ex["question"]
        opts = f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}"
        gold_idx = int(ex["cop"])
        msgs = [
            {"role": "system", "content": "You are a board-certified physician. Answer the multiple-choice medical exam question strictly with one letter: A, B, C, or D."},
            {"role": "user", "content": f"Question: {q}\n{opts}\nCorrect Option (A, B, C, or D):"}
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=768, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1, :]
            scores = [float(lg[cid].item()) for cid in opt_ids]
        if scores.index(max(scores)) == gold_idx:
            mcqa_ok += 1
    mcqa_acc = round(mcqa_ok / 200 * 100.0, 2)
    print(f"  [MedMCQA 4-Choice n=200] Accuracy: {mcqa_acc}% ({mcqa_ok}/200)")

    res = {
        "model": "MedVision-7B-v2-MultiTask (12k samples, 2.6M tokens, Val PPL 1.89)",
        "pubmedqa_unseen_200_acc_pct": pqa_acc,
        "pubmedqa_preds": dict(Counter(pqa_preds)),
        "medmcqa_4choice_200_acc_pct": mcqa_acc,
    }
    with open("./checkpoints/phase2b_dual_benchmark.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  MEDVISION-7B-v2 DUAL BENCHMARK SUMMARY")
    print(f"  1. PubMedQA (Unseen n=200)  : {pqa_acc}%  (Human Expert Avg: 78.0%)")
    print(f"  2. MedMCQA 4-Choice (n=200) : {mcqa_acc}%  (Zero-Shot Medical Board Exam)")
    print("=" * 80)

if __name__ == "__main__":
    main()
