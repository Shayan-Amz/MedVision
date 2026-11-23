import json, math, os, time, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

class MetricsLogger(TrainerCallback):
    def __init__(self):
        self.train_logs, self.eval_logs = [], []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        if "loss" in logs:
            self.train_logs.append({"step": state.global_step, "loss": round(float(logs["loss"]), 4), "vram_gb": vram})
        if "eval_loss" in logs:
            el = float(logs["eval_loss"])
            self.eval_logs.append({"step": state.global_step, "eval_loss": round(el, 4), "perplexity": round(math.exp(min(el, 20.0)), 2)})

def main():
    assert torch.cuda.is_available(), "CUDA GPU required! Activate .venv_gpu first."
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    out_dir = "./checkpoints/medvision-7b-v2-multitask-sft"
    os.makedirs("./checkpoints", exist_ok=True)

    print("=" * 84)
    print("  PHASE 2A: Scaled Multi-Task Medical SFT (12,000 Train / 500 Held-Out Val)")
    print("  Sources: Medical Flashcards (8k) + PubMedQA Artificial (3.2k) + Labeled (800)")
    print("=" * 84)

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Source 1: 8,000 Train + 500 Held-Out Val from Medical Flashcards
    fc_raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train").shuffle(seed=42).select(range(8500))
    def fmt_fc(ex):
        msgs = [
            {"role": "system", "content": "You are MedVision Clinical AI, a board-certified medical assistant providing accurate, concise, evidence-based clinical reasoning."},
            {"role": "user", "content": (ex.get("input") or "").strip()},
            {"role": "assistant", "content": (ex.get("output") or "").strip()}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}
    fc_fmt = fc_raw.map(fmt_fc, remove_columns=fc_raw.column_names)
    fc_train, fc_val = fc_fmt.select(range(8000)), fc_fmt.select(range(8000, 8500))

    # Source 2: 3,200 PubMedQA Artificial (research abstract -> clinical decision)
    art_raw = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=42).select(range(3200))
    def fmt_pqa(ex):
        ctx = " ".join(ex["context"]["contexts"])[-1600:]
        q = ex["question"]
        ans = ex["final_decision"].strip().capitalize()
        msgs = [
            {"role": "system", "content": "You are a medical expert. Answer the clinical question strictly with one word: Yes, No, or Maybe."},
            {"role": "user", "content": f"Abstract:\n{ctx}\n\nQuestion: {q}\nAnswer (Yes, No, or Maybe):"},
            {"role": "assistant", "content": ans}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}
    art_train = art_raw.map(fmt_pqa, remove_columns=art_raw.column_names)

    # Source 3: 800 PubMedQA Labeled Train Split (indices 200..1000; 0..200 strictly UNSEEN!)
    lbl_raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42).select(range(200, 1000))
    lbl_train = lbl_raw.map(fmt_pqa, remove_columns=lbl_raw.column_names)

    train_ds = concatenate_datasets([fc_train, art_train, lbl_train]).shuffle(seed=42)
    print(f"[DATASET READY] Total Multi-Task Train: {len(train_ds)} | Held-Out Val: {len(fc_val)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    logger = MetricsLogger()
    sft_args = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1.2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=fc_val, processing_class=tok, peft_config=lora_cfg, callbacks=[logger])
    t0 = time.perf_counter()
    trainer.train()
    ev = trainer.evaluate()
    elapsed = round(time.perf_counter() - t0, 1)
    trainer.save_model(out_dir)

    val_loss = round(float(ev["eval_loss"]), 4)
    val_ppl = round(math.exp(min(val_loss, 20.0)), 2)
    report = {
        "train_samples": len(train_ds),
        "val_samples": len(fc_val),
        "elapsed_sec": elapsed,
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
        "train_logs": logger.train_logs,
        "eval_logs": logger.eval_logs,
    }
    with open("./checkpoints/phase2a_sft_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 84)
    print(f"  PHASE 2A COMPLETE in {elapsed}s! (12,000 Multi-Task Medical Samples)")
    print(f"  Held-Out Val Loss: {val_loss} | Val Perplexity: {val_ppl}")
    print(f"  Saved Checkpoint : {os.path.abspath(out_dir)}")
    print("=" * 84)

if __name__ == "__main__":
    main()
