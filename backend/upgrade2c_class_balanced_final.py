import json, os, time, torch
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def compute_macro_f1(golds, preds, labels=("yes", "no", "maybe")):
    f1s = {}
    for lbl in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(golds, preds) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(golds, preds) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[lbl] = round(2 * prec * rec / (prec + rec) * 100, 2) if (prec + rec) > 0 else 0.0
    macro = round(sum(f1s.values()) / len(labels), 2)
    return macro, f1s

def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    prev_adapter = "./checkpoints/medvision-7b-stage2-dpo-final"
    final_adapter = "./checkpoints/medvision-7b-clinical-balanced-final"

    print("=" * 82)
    print("  UPGRADE 2C: Class-Balanced PubMedQA Fine-Tuning (Solving Minority 'maybe' Collapse)")
    print("=" * 82)

    full_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
    test_ds = full_ds.select(range(200))
    train_raw = full_ds.select(range(200, 1000))

    # Class-Balanced Oversampling on TRAIN split ONLY (Zero Data Leakage!)
    maybe_only = train_raw.filter(lambda x: x["final_decision"].strip().lower() == "maybe")
    train_balanced_raw = concatenate_datasets([train_raw, maybe_only, maybe_only]).shuffle(seed=42)
    print(f"[CLASS BALANCING] Original Train: {len(train_raw)} -> Balanced Train: {len(train_balanced_raw)} (Oversampled 'maybe' x3)")

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

    train_ds = train_balanced_raw.map(fmt_train, remove_columns=train_balanced_raw.column_names)

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
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, processing_class=tok)
    t0 = time.perf_counter()
    trainer.train()
    train_sec = round(time.perf_counter() - t0, 1)
    trainer.save_model(final_adapter)
    print(f"\n[BALANCED TRAIN COMPLETE] {train_sec}s | Evaluating on 200 UNSEEN Test Questions...")

    model.eval()
    choice_ids = {
        "yes": tok.encode("Yes", add_special_tokens=False)[0],
        "no": tok.encode("No", add_special_tokens=False)[0],
        "maybe": tok.encode("Maybe", add_special_tokens=False)[0],
    }

    golds, preds = [], []
    for idx, ex in enumerate(test_ds):
        ctx = " ".join(ex["context"]["contexts"])[-1800:]
        q = ex["question"]
        gold = ex["final_decision"].strip().lower()
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
        golds.append(gold)
        preds.append(pred)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_ds):
            acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100
            print(f"  [Balanced-7B] {idx+1}/200 | Accuracy: {acc:.1f}% | Preds: {dict(Counter(preds))}")

    final_acc = round(sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) * 100.0, 2)
    macro_f1, per_class_f1 = compute_macro_f1(golds, preds)

    out = {
        "benchmark": "qiaojin/PubMedQA (pqa_labeled, Class-Balanced Train / 200 Unseen Test)",
        "base_accuracy_pct": 51.5,
        "unbalanced_sft_accuracy_pct": 77.0,
        "medvision_7b_sft_dpo_accuracy_pct": final_acc,
        "delta_pct": round(final_acc - 51.5, 2),
        "macro_f1_pct": macro_f1,
        "per_class_f1_pct": per_class_f1,
        "gold_distribution": dict(Counter(golds)),
        "aligned_pred_distribution": dict(Counter(preds)),
        "final_checkpoint": os.path.abspath(final_adapter),
    }
    with open("./checkpoints/upgrade2_pubmedqa_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 82)
    print(f"  Base Qwen2.5-7B-Instruct (Calibrated)  : 51.5%")
    print(f"  MedVision-7B Balanced (Unseen n=200)   : {final_acc}%  [Delta: +{final_acc - 51.5:.2f}%]")
    print(f"  Macro-F1 Across All 3 Classes          : {macro_f1}%  | Per-Class F1: {per_class_f1}")
    print(f"  Predicted Class Distribution           : {dict(Counter(preds))}")
    print("=" * 82)

if __name__ == "__main__":
    main()
