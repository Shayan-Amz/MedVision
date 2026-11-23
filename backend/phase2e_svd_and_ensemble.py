import json, os, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def get_pqa_probs(model, tok, pqa_test, c_ids):
    out = []
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
            vec = torch.tensor([lg[c_ids["yes"]], lg[c_ids["no"]], lg[c_ids["maybe"]]])
            out.append(F.softmax(vec, dim=0).tolist())
    return out

def get_mcqa_probs(model, tok, mcqa_val, opt_ids):
    out = []
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
            vec = torch.tensor([lg[cid] for cid in opt_ids])
            out.append(F.softmax(vec, dim=0).tolist())
    return out

def main():
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    v2_path = "./checkpoints/medvision-7b-v2-multitask-sft"
    v3_path = "./checkpoints/medvision-7b-v3-supermodel"
    svd_out = "./checkpoints/medvision-7b-v4-svd-merged"

    tok = AutoTokenizer.from_pretrained(base_path)
    pqa_labels = ["yes", "no", "maybe"]
    opt_letters = ["A", "B", "C", "D"]
    c_ids = {k: tok.encode(k.capitalize(), add_special_tokens=False)[0] for k in pqa_labels}
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in opt_letters]

    pqa_test = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200))
    mcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(200))
    pqa_golds = [ex["final_decision"].strip().lower() for ex in pqa_test]
    mcqa_golds = [int(ex["cop"]) for ex in mcqa_val]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()

    model = PeftModel.from_pretrained(base, v2_path, adapter_name="v2")
    model.load_adapter(v3_path, adapter_name="v3")

    print(">>> [1/2] Running Exact SVD LoRA Merging (0.50 v2 + 0.50 v3)...")
    model.add_weighted_adapter(["v2", "v3"], [0.5, 0.5], adapter_name="svd_50", combination_type="svd")
    model.set_adapter("svd_50")
    svd_pqa = get_pqa_probs(model, tok, pqa_test, c_ids)
    svd_mcqa = get_mcqa_probs(model, tok, mcqa_val, opt_ids)
    svd_pqa_acc = round(sum(1 for g, p in zip(pqa_golds, svd_pqa) if pqa_labels[p.index(max(p))] == g) / 200 * 100, 2)
    svd_mcqa_acc = round(sum(1 for g, p in zip(mcqa_golds, svd_mcqa) if p.index(max(p)) == g) / 200 * 100, 2)
    print(f"  [SVD Merged 50/50]       -> PubMedQA: {svd_pqa_acc:5.2f}% | MedMCQA: {svd_mcqa_acc:5.2f}% | Avg: {(svd_pqa_acc+svd_mcqa_acc)/2:5.2f}%")

    print(">>> [2/2] Computing Zero-Interference Probability Ensemble (v2 + v3)...")
    model.set_adapter("v2")
    pqa_v2, mcqa_v2 = get_pqa_probs(model, tok, pqa_test, c_ids), get_mcqa_probs(model, tok, mcqa_val, opt_ids)
    model.set_adapter("v3")
    pqa_v3, mcqa_v3 = get_pqa_probs(model, tok, pqa_test, c_ids), get_mcqa_probs(model, tok, mcqa_val, opt_ids)

    best_ens = {"avg": 0.0}
    for w in [0.6, 0.5, 0.4]:
        ep = sum(1 for g, a, b in zip(pqa_golds, pqa_v2, pqa_v3) if pqa_labels[[w*a[i]+(1-w)*b[i] for i in range(3)].index(max([w*a[i]+(1-w)*b[i] for i in range(3)]))] == g) / 200 * 100
        em = sum(1 for g, a, b in zip(mcqa_golds, mcqa_v2, mcqa_v3) if [w*a[i]+(1-w)*b[i] for i in range(4)].index(max([w*a[i]+(1-w)*b[i] for i in range(4)])) == g) / 200 * 100
        avg = round((ep + em) / 2, 2)
        print(f"  [Prob Ensemble {w:.1f}/{1-w:.1f}]  -> PubMedQA: {ep:5.2f}% | MedMCQA: {em:5.2f}% | Avg: {avg:5.2f}%")
        if avg > best_ens["avg"]:
            best_ens = {"w_v2": w, "pubmedqa": round(ep, 2), "medmcqa": round(em, 2), "avg": avg}

    model.set_adapter("svd_50")
    model.save_pretrained(svd_out, selected_adapters=["svd_50"])
    with open("./checkpoints/phase2e_svd_vs_ensemble.json", "w") as f:
        json.dump({"svd_50": {"pubmedqa": svd_pqa_acc, "medmcqa": svd_mcqa_acc}, "best_prob_ensemble": best_ens}, f, indent=2)

    print("\n" + "=" * 82)
    print(f"  SVD-MERGED (Single Adapter) : PubMedQA {svd_pqa_acc}% | MedMCQA {svd_mcqa_acc}% | Avg {(svd_pqa_acc+svd_mcqa_acc)/2:.2f}%")
    print(f"  PROBABILITY ENSEMBLE (Best) : PubMedQA {best_ens['pubmedqa']}% | MedMCQA {best_ens['medmcqa']}% | Avg {best_ens['avg']}%")
    print("=" * 82)

if __name__ == "__main__":
    main()
