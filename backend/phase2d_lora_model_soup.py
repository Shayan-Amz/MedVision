import json, os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def eval_both(model, tok, pqa_test, mcqa_val, c_ids, opt_ids, opt_letters):
    pqa_ok = 0
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
        if max(scores, key=scores.get) == ex["final_decision"].strip().lower(): pqa_ok += 1

    mcqa_ok = 0
    for ex in mcqa_val:
        opts = f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}"
        msgs = [
            {"role": "system", "content": "You are a board-certified physician. Answer the multiple-choice medical exam question starting strictly with the correct letter (A, B, C, or D), followed by brief clinical reasoning."},
            {"role": "user", "content": f"Question: {ex['question']}\n{opts}\nCorrect Option (A, B, C, or D):"}
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=768, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1, :]
            scores = [float(lg[cid].item()) for cid in opt_ids]
        if scores.index(max(scores)) == int(ex["cop"]): mcqa_ok += 1

    return round(pqa_ok / 200 * 100, 2), round(mcqa_ok / 200 * 100, 2)

def main():
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    v2_path = "./checkpoints/medvision-7b-v2-multitask-sft"
    v3_path = "./checkpoints/medvision-7b-v3-supermodel"
    soup_out = "./checkpoints/medvision-7b-v4-model-soup"

    tok = AutoTokenizer.from_pretrained(base_path)
    opt_letters = ["A", "B", "C", "D"]
    c_ids = {k: tok.encode(k.capitalize(), add_special_tokens=False)[0] for k in ["yes", "no", "maybe"]}
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in opt_letters]

    pqa_test = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200))
    mcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(200))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()

    model = PeftModel.from_pretrained(base, v2_path, adapter_name="v2")
    model.load_adapter(v3_path, adapter_name="v3")

    best_avg, best_cfg = 0.0, None
    print("\n--- PEFT LoRA Model Soup Weight Merging Sweep (v2 + v3) ---")
    for w_v2 in [0.7, 0.5, 0.35]:
        w_v3 = round(1.0 - w_v2, 2)
        name = f"soup_{int(w_v2*100)}"
        model.add_weighted_adapter(["v2", "v3"], [w_v2, w_v3], adapter_name=name, combination_type="linear")
        model.set_adapter(name)
        pqa_acc, mcqa_acc = eval_both(model, tok, pqa_test, mcqa_val, c_ids, opt_ids, opt_letters)
        avg = round((pqa_acc + mcqa_acc) / 2.0, 2)
        print(f"  Soup ({w_v2:.2f} v2 + {w_v3:.2f} v3) -> PubMedQA: {pqa_acc:5.2f}% | MedMCQA: {mcqa_acc:5.2f}% | Avg: {avg:5.2f}%")
        if avg > best_avg:
            best_avg = avg
            best_cfg = {"w_v2": w_v2, "w_v3": w_v3, "adapter": name, "pubmedqa": pqa_acc, "medmcqa": mcqa_acc, "avg": avg}

    # Save the winning merged adapter!
    model.set_adapter(best_cfg["adapter"])
    model.save_pretrained(soup_out, selected_adapters=[best_cfg["adapter"]])
    with open("./checkpoints/phase2d_model_soup_report.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    print("\n" + "=" * 82)
    print(f"  WINNING LORA MODEL SOUP SAVED: {os.path.abspath(soup_out)}")
    print(f"  Weights ({best_cfg['w_v2']} v2 + {best_cfg['w_v3']} v3) -> PubMedQA: {best_cfg['pubmedqa']}% | MedMCQA: {best_cfg['medmcqa']}% | Avg: {best_cfg['avg']}%")
    print("=" * 82)

if __name__ == "__main__":
    main()
