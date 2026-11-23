"""
MedVision & Clinical LLM Studio — Research-Grade 4-bit QLoRA Fine-Tuning on NVIDIA RTX 4080 (12GB)
Trains Qwen2.5-3B-Instruct (or Qwen2.5-7B-Instruct) on a REAL Hugging Face Clinical Dataset
(1,000 Train / 200 Held-Out Validation split) with:
  - 4-bit NormalFloat (NF4) Double Quantization via bitsandbytes
  - Ada Lovelace bfloat16 mixed precision + Paged AdamW 8-bit
  - Real Train Loss vs. Held-Out Validation Loss & Perplexity (Zero Overfitting Audit)
  - Peak GPU VRAM telemetry & Real Saved Adapter Checkpoint
"""
import json
import math
import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


class GPUMetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
        if "loss" in logs:
            self.train_logs.append({
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0.0), 2),
                "train_loss": round(float(logs["loss"]), 4),
                "vram_peak_gb": vram_gb,
            })
        if "eval_loss" in logs:
            ev_loss = float(logs["eval_loss"])
            self.eval_logs.append({
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0.0), 2),
                "eval_loss": round(ev_loss, 4),
                "perplexity": round(math.exp(min(ev_loss, 20.0)), 2),
                "vram_peak_gb": vram_gb,
            })


def load_real_medical_splits(train_size: int = 1000, val_size: int = 200):
    """Loads real clinical QA pairs from Hugging Face and formats them into ChatML clinical prompts."""
    print(f"[DATASET] Loading real medical dataset from Hugging Face ({train_size} train / {val_size} val)...")
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    ds = ds.shuffle(seed=42).select(range(train_size + val_size))

    def format_clinical_example(ex):
        q = (ex.get("input") or "").strip()
        a = (ex.get("output") or "").strip()
        return {
            "text": (
                "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
                "accurate, concise, evidence-based clinical reasoning.<|im_end|>\n"
                f"<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n{a}<|im_end|>"
            )
        }

    ds = ds.map(format_clinical_example, remove_columns=ds.column_names)
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, train_size + val_size))
    return train_ds, val_ds


def main():
    assert torch.cuda.is_available(), "CUDA GPU not detected! Activate .venv_gpu first."

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Default: Qwen2.5-3B-Instruct (~6-8 min on RTX 4080). Set MEDVISION_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" for 7B!
    model_id = os.environ.get("MEDVISION_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    output_dir = "./checkpoints/medvision-rtx4080-medical-qlora"
    os.makedirs("./checkpoints", exist_ok=True)

    print("=" * 80)
    print("  MedVision AI Studio — REAL 4-bit QLoRA Medical Training on NVIDIA RTX 4080")
    print("=" * 80)
    print(f"[GPU]        : {gpu_name} ({vram_total:.2f} GB VRAM)")
    print(f"[BASE MODEL] : {model_id} (4-bit NF4 + BF16 Compute)")

    train_ds, val_ds = load_real_medical_splits(train_size=1000, val_size=200)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    metrics_cb = GPUMetricsCallback()

    sft_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[metrics_cb],
    )

    print("\n>>> Evaluating Baseline Perplexity on Held-Out 200-Sample Medical Validation Set...")
    baseline_eval = trainer.evaluate()
    base_loss = round(float(baseline_eval["eval_loss"]), 4)
    base_ppl = round(math.exp(min(base_loss, 20.0)), 2)
    print(f"[BASELINE] Held-Out Val Loss: {base_loss} | Baseline Perplexity: {base_ppl}")

    print("\n>>> Launching 4-bit QLoRA Training on NVIDIA RTX 4080...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed_sec = round(time.perf_counter() - t0, 1)

    print("\n>>> Evaluating Fine-Tuned Adapter on Held-Out 200-Sample Medical Validation Set...")
    final_eval = trainer.evaluate()
    final_loss = round(float(final_eval["eval_loss"]), 4)
    final_ppl = round(math.exp(min(final_loss, 20.0)), 2)
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    trainer.save_model(output_dir)

    report = {
        "hardware": gpu_name,
        "base_model": model_id,
        "quantization": "4-bit NF4 Double-Quant + BF16 (bitsandbytes)",
        "lora_config": {"r": 16, "alpha": 32, "target_modules": "all_linear_7"},
        "dataset": "medalpaca/medical_meadow_medical_flashcards (1000 Train / 200 Held-Out Val)",
        "elapsed_seconds": elapsed_sec,
        "peak_vram_gb": peak_vram_gb,
        "baseline_val_loss": base_loss,
        "baseline_perplexity": base_ppl,
        "finetuned_val_loss": final_loss,
        "finetuned_perplexity": final_ppl,
        "perplexity_reduction_pct": round((base_ppl - final_ppl) / max(base_ppl, 1e-6) * 100.0, 2),
        "train_curve": metrics_cb.train_logs,
        "eval_curve": metrics_cb.eval_logs,
    }

    report_path = "./checkpoints/rtx4080_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  RTX 4080 4-BIT QLORA TRAINING COMPLETE IN {elapsed_sec}s!")
    print(f"  Baseline Val Perplexity  : {base_ppl}  (Loss: {base_loss})")
    print(f"  Fine-Tuned Val Perplexity: {final_ppl}  (Loss: {final_loss})  [-{report['perplexity_reduction_pct']}%]")
    print(f"  Peak RTX 4080 VRAM Used  : {peak_vram_gb} GB / {vram_total:.1f} GB")
    print(f"  Saved Adapter Checkpoint : {os.path.abspath(output_dir)}")
    print(f"  Saved Research Report    : {os.path.abspath(report_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
