import json, os, time, torch
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    assert torch.cuda.is_available(), "CUDA GPU required! Activate .venv_gpu first."
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    prev_adapter = "./checkpoints/medvision-7b-v2-multitask-sft"
    out_dir = "./checkpoints/medvision-7b-v3-supermodel"
    os.makedirs("./checkpoints", exist_ok=True)

    print("=" * 84)
    print("  PHASE 2C: MedVision-7B-v3 SuperModel (10,000 MedMCQA + 2,000 PubMedQA Replay)")
    print("  Goal: Push MedMCQA > 62-68% AND Push PubMedQA >= 78-80% (Human Expert Level!)")
    print("=" * 84)

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # 1. 10,000 MedMCQA Train questions (strictly from 'train' split; 'validation' is 100% UNSEEN!)
    mcqa_raw = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=42).select(range(10000))
    opt_letters = ["A", "B", "C", "D"]
    def fmt_mcqa(ex):
        q = (ex.get("question") or "").strip()
        opts = f"A) {ex['opa']}\nB) {ex['opb']}\nC) {ex['opc']}\nD) {ex['opd']}"
        ans_letter = opt_letters[int(ex["cop"])] if int(ex["cop"]) in (0, 1, 2, 3) else "A"
        exp = (ex.get("exp") or "").strip()
        # Keep answer letter first so log-likelihood scoring & clinical reasoning both improve!
        resp = f"{ans_letter}. {exp[:280]}" if len(exp) > 20 else ans_letter
        msgs = [
            {"role": "system", "content": "You are a board-certified physician. Answer the multiple-choice medical exam question starting strictly with the correct letter (A, B, C, or D), followed by brief clinical reasoning."},
            {"role": "user", "content": f"Question: {q}\n{opts}\nCorrect Option (A, B, C, or D):"},
            {"role": "assistant", "content": resp}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}
    mcqa_train = mcqa_raw.map(fmt_mcqa, remove_columns=mcqa_raw.column_names)

    # 2. 2,000 PubMedQA Replay Buffer (1,200 artificial + 800 labeled train [200..1000], 0..200 strictly UNSEEN!)
    def fmt_pqa(ex):
        ctx = " ".join(ex["context"]["contexts"])[-1600:]
        ans = ex["final_decision"].strip().capitalize()
        msgs = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer (Yes, No, or Maybe):"},
            {"role": "assistant", "content": ans}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    pqa_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=99).select(range(1200)).map(fmt_pqa, remove_columns=["pubid", "question", "context", "long_answer", "final_decision"])
    pqa_lbl = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200, 1000)).map(fmt_pqa, remove_columns=["pubid", "question", "context", "long_answer", "final_decision"])

    train_ds = concatenate_datasets([mcqa_train, pqa_art, pqa_lbl]).shuffle(seed=42)
    print(f"[DATASET READY] Total Mixed Replay Train: {len(train_ds)} samples (0% overlap with test sets)")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, prev_adapter, is_trainable=True)

    sft_args = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=6e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, processing_class=tok)
    t0 = time.perf_counter()
    trainer.train()
    elapsed = round(time.perf_counter() - t0, 1)
    trainer.save_model(out_dir)
    print(f"\n[TRAINING COMPLETE in {elapsed}s] Running Final Dual Benchmark on UNSEEN Test Sets...")

    model.eval()
    # Eval 1: PubMedQA (Exact same 200 UNSEEN)
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
        if pred == ex["final_decision"].strip().lower(): pqa_ok += 1
    pqa_acc = round(pqa_ok / 200 * 100.0, 2)

    # Eval 2: MedMCQA Validation (Exact same 200 UNSEEN)
    mcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation").shuffle(seed=42).select(range(200))
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in opt_letters]
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
    mcqa_acc = round(mcqa_ok / 200 * 100.0, 2)

    summary = {
        "model": "MedVision-7B-v3-SuperModel (24k Cumulative Medical Samples)",
        "elapsed_sec": elapsed,
        "pubmedqa_unseen_200_acc_pct": pqa_acc,
        "pubmedqa_preds": dict(Counter(pqa_preds)),
        "medmcqa_unseen_200_acc_pct": mcqa_acc,
        "checkpoint": os.path.abspath(out_dir),
    }
    with open("./checkpoints/phase2c_supermodel_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 84)
    print(f"  MEDVISION-7B-v3 SUPERMODEL FINAL BENCHMARK (UNSEEN TEST SETS)")
    print(f"  1. PubMedQA (n=200) : {pqa_acc}%  (Prev v2: 77.5% | Human Expert: 78.0%)")
    print(f"  2. MedMCQA  (n=200) : {mcqa_acc}%  (Prev v2: 52.0% | Random: 25.0%)")
    print(f"  Saved Checkpoint    : {os.path.abspath(out_dir)}")
    print("=" * 84)

if __name__ == "__main__":
    main()
