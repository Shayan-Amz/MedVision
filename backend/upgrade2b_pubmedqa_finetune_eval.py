import gc, json, os, time, torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    prev_adapter = "./checkpoints/medvision-7b-stage2-dpo-final"
    final_adapter = "./checkpoints/medvision-7b-clinical-final"

    print("=" * 80)
    print("  UPGRADE 2B: PubMedQA Clinical Reasoning Fine-Tuning (800 Train / 200 Held-Out Test)")
    print("=" * 80)

    # Strict 0% Data Leakage Split: First 200 = Held-Out Test (exact same 200 as Baseline 51.5%)
    # Remaining 800 = Train Split
    full_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
    test_ds = full_ds.select(range(200))
    train_raw = full_ds.select(range(200, 1000))

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt_train(ex):
        ctx = " ".join(ex["context"]["contexts"])[-1800:]
        q = ex["question"]
        ans = ex["final_decision"].strip().capitalize()
        messages = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {q}\nAnswer (Yes, No, or Maybe):"},
            {"role": "assistant", "content": ans}
        ]
        return {"text": tok.apply_chat_template(messages, tokenize=False)}

    train_ds = train_raw.map(fmt_train, remove_columns=train_raw.column_names)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, prev_adapter, is_trainable=True)

    sft_args = SFTConfig(
        output_dir=final_adapter,
        dataset_text_field="text",
        max_length=768,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tok,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_sec = round(time.perf_counter() - t0, 1)
    trainer.save_model(final_adapter)
    print(f"\n[TRAIN COMPLETE] {train_sec}s | Evaluating on 200 UNSEEN Held-Out Test Questions...")

    model.eval()
    choice_ids = {
        "yes": tok.encode("Yes", add_special_tokens=False)[0],
        "no": tok.encode("No", add_special_tokens=False)[0],
        "maybe": tok.encode("Maybe", add_special_tokens=False)[0],
    }

    correct, total = 0, len(test_ds)
    pred_counts = Counter()
    gold_counts = Counter()
    for idx, ex in enumerate(test_ds):
        ctx = " ".join(ex["context"]["contexts"])[-1800:]
        q = ex["question"]
        gold = ex["final_decision"].strip().lower()
        gold_counts[gold] += 1
        messages = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {q}\nAnswer (Yes, No, or Maybe):"}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            scores = {k: float(logits[v].item()) for k, v in choice_ids.items()}
        pred = max(scores, key=scores.get)
        pred_counts[pred] += 1
        if pred == gold:
            correct += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [MedVision-7B-Final] {idx+1}/{total} | Held-Out Test Accuracy: {correct/(idx+1)*100:.1f}% | Preds: {dict(pred_counts)}")

    final_acc = round(correct / total * 100.0, 2)
    base_acc = 51.5
    delta = round(final_acc - base_acc, 2)
    out = {
        "benchmark": "qiaojin/PubMedQA (pqa_labeled, 800 Train / 200 Unseen Held-Out Test)",
        "base_accuracy_pct": base_acc,
        "medvision_7b_sft_dpo_accuracy_pct": final_acc,
        "delta_pct": delta,
        "gold_distribution": dict(gold_counts),
        "aligned_pred_distribution": dict(pred_counts),
        "final_checkpoint": os.path.abspath(final_adapter),
    }
    with open("./checkpoints/upgrade2_pubmedqa_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  Base Qwen2.5-7B-Instruct (Calibrated Baseline) : {base_acc}% (103/200)")
    print(f"  MedVision-7B-Clinical-Final (Unseen Test Set)  : {final_acc}% ({correct}/200)  [Delta: {delta:+.2f}%]")
    print("=" * 80)

if __name__ == "__main__":
    main()
