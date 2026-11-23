import json, os, re, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def extract_choice(text: str) -> str:
    m = re.search(r"Final Answer\s*[:\-]?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.findall(r"\b([ABCD])\b", text)
    return m2[-1].upper() if m2 else "A"

def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    adapter_path = "./checkpoints/medvision-7b-v4-cat-merged/cat_50"
    n = 200
    batch_size = 8

    print("=" * 80)
    print(f"  LEVER 1: Batched Chain-of-Thought (Reason-First) on MedMCQA (n={n})")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(base_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()
    model = PeftModel.from_pretrained(base, adapter_path).eval()

    mcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(n))
    opt_letters = ["A", "B", "C", "D"]

    prompts, golds = [], []
    for ex in mcqa_val:
        opts = f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}"
        msgs = [
            {"role": "system", "content": "You are a board-certified physician. Think step-by-step in 2-3 brief sentences to eliminate wrong choices, then end on a new line with strictly: Final Answer: X (where X is A, B, C, or D)."},
            {"role": "user", "content": f"Question: {ex['question']}\n{opts}\nReason briefly, then give Final Answer:"}
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        golds.append(opt_letters[int(ex["cop"])])

    correct = 0
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_golds = golds[start:start + batch_size]
        inp = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=640, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outs = model.generate(
                **inp,
                max_new_tokens=110,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen_tokens = outs[:, inp["input_ids"].shape[1]:]
        decoded = tok.batch_decode(gen_tokens, skip_special_tokens=True)
        for text, gold in zip(decoded, batch_golds):
            if extract_choice(text) == gold:
                correct += 1
        done = min(start + batch_size, n)
        if done % 16 == 0 or done == n:
            print(f"  [CoT MedMCQA] {done}/{n} | Running Accuracy: {correct/done*100:.1f}% ({time.perf_counter()-t0:.0f}s)", flush=True)

    acc = round(correct / n * 100.0, 2)
    elapsed = round(time.perf_counter() - t0, 1)
    with open("./checkpoints/phase2f_cot_medmcqa.json", "w") as f:
        json.dump({"method": "Chain-of-Thought (Reason-First)", "model": "MedVision-7B-v4-CAT", "n": n, "accuracy_pct": acc, "prev_no_cot_pct": 55.0, "delta_pct": round(acc - 55.0, 2), "elapsed_sec": elapsed}, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  PREVIOUS (1-Token Direct, No CoT) : 55.00% (110/200)")
    print(f"  NEW (Chain-of-Thought Reasoning)  : {acc:.2f}% ({correct}/{n})  [Delta: {acc-55.0:+.2f}%] in {elapsed}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
