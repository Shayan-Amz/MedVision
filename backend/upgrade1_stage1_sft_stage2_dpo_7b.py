"""
MedVision & Clinical LLM Studio — UPGRADE 1: Two-Stage Research Pipeline on NVIDIA RTX 4080 (12GB)
  - Stage 1 (Scaled QLoRA SFT): Qwen2.5-7B-Instruct on 3,000 real medical QA samples (300 Held-Out Val)
    with LoRA r=32, alpha=64, 4-bit NF4, Ada Lovelace BF16.
  - Stage 2 (Clinical DPO Alignment): Loads the Stage-1 SFT adapter and aligns it via Direct Preference
    Optimization (DPO) on curated clinical safety/contraindication preference pairs.
Outputs:
  - ./checkpoints/medvision-7b-stage1-sft/
  - ./checkpoints/medvision-7b-stage2-dpo-final/
  - ./checkpoints/upgrade1_two_stage_report.json
"""
import gc
import json
import math
import os
import time
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig


CLINICAL_SAFETY_DPO_PAIRS = [
    {
        "prompt": (
            "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
            "accurate, concise, contraindication-aware clinical reasoning.<|im_end|>\n"
            "<|im_start|>user\nChest X-Ray shows Right Lower Lobe Pneumonia (91.4%, Grad-CAM ROI [0.56, 0.48, 0.84, 0.79]). "
            "Patient: 58M, SpO2 91%, Temp 38.9C, Type 2 Diabetes (HbA1c 7.8%). What is your impression and protocol?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "chosen": (
            "IMPRESSION: Acute right lower lobe lobar pneumonia (91.4%, ROI [0.56, 0.48, 0.84, 0.79]) with hypoxemia (SpO2 91%). "
            "PROTOCOL: Initiate empiric IV Ceftriaxone 2g q24h + Azithromycin 500mg daily. Strictly avoid systemic corticosteroids "
            "due to comorbid Type 2 Diabetes unless refractory septic shock. Obtain blood cultures x2 and maintain SpO2 >= 94%."
        ),
        "rejected": (
            "There is a white patch in the lung that looks like pneumonia. Give standard antibiotics and Prednisone 40mg steroids to reduce inflammation."
        ),
    },
    {
        "prompt": (
            "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
            "accurate, concise, contraindication-aware clinical reasoning.<|im_end|>\n"
            "<|im_start|>user\nBrain MRI T1+C shows a 3.8cm left frontotemporal ring-enhancing mass (94.2%) with 4mm midline shift. "
            "Patient: 46F, GCS 14/15, expressive aphasia. Next steps?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "chosen": (
            "IMPRESSION: High-grade primary glial neoplasm (WHO Grade 4 glioblastoma, 94.2%) with vasogenic edema and 4mm rightward midline shift near Broca's area. "
            "PROTOCOL: Administer STAT IV Dexamethasone 10mg loading dose, initiate Levetiracetam 500mg BID seizure prophylaxis, and request emergent Neurosurgery consult."
        ),
        "rejected": (
            "Brain scan shows a mass on the left side. Recommend outpatient follow-up MRI in 3 months and over-the-counter painkillers."
        ),
    },
    {
        "prompt": (
            "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
            "accurate, concise, contraindication-aware clinical reasoning.<|im_end|>\n"
            "<|im_start|>user\nChest X-Ray shows bilateral bat-wing pulmonary edema (89.5%). "
            "Patient: 74F, SpO2 88%, Chronic Kidney Disease Stage 3b (eGFR 36). Next steps?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "chosen": (
            "IMPRESSION: Acute cardiogenic pulmonary edema (89.5%) with severe hypoxemia (SpO2 88%). "
            "PROTOCOL: Initiate NIV/CPAP and IV Furosemide diuresis with strict urine output and serum creatinine monitoring given CKD Stage 3b (eGFR 36). Avoid nephrotoxic IV contrast."
        ),
        "rejected": (
            "Fluid in both lungs. Perform high-dose contrast CT angiography immediately and advise oral hydration."
        ),
    },
    {
        "prompt": (
            "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
            "accurate, concise, contraindication-aware clinical reasoning.<|im_end|>\n"
            "<|im_start|>user\nChest X-Ray shows left-sided tension pneumothorax (96.8%) with tracheal deviation. "
            "Patient: 32M, BP 82/50, HR 132. Immediate management?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "chosen": (
            "IMPRESSION: CRITICAL left-sided tension pneumothorax (96.8%) with obstructive shock (BP 82/50). "
            "PROTOCOL: Immediate needle decompression (4th/5th intercostal space, anterior axillary line) followed by emergent tube thoracostomy. Do not delay for CT imaging."
        ),
        "rejected": (
            "Air in the left chest cavity. Schedule a routine Chest CT scan tomorrow morning to confirm the size."
        ),
    },
] * 8  # 32 high-signal clinical safety DPO preference pairs


class MetricsLogger(TrainerCallback):
    def __init__(self):
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
        if "loss" in logs:
            self.train_logs.append({
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0.0), 2),
                "loss": round(float(logs["loss"]), 4),
                "vram_gb": vram,
            })
        if "eval_loss" in logs:
            el = float(logs["eval_loss"])
            self.eval_logs.append({
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0.0), 2),
                "eval_loss": round(el, 4),
                "perplexity": round(math.exp(min(el, 20.0)), 2),
                "vram_gb": vram,
            })


def main():
    assert torch.cuda.is_available(), "CUDA GPU required! Activate .venv_gpu first."
    base_model_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    train_size = int(os.environ.get("MEDVISION_TRAIN_SIZE", "3000"))
    val_size = int(os.environ.get("MEDVISION_VAL_SIZE", "300"))

    stage1_dir = "./checkpoints/medvision-7b-stage1-sft"
    stage2_dir = "./checkpoints/medvision-7b-stage2-dpo-final"
    os.makedirs("./checkpoints", exist_ok=True)

    print("=" * 82)
    print(f"  UPGRADE 1: Two-Stage QLoRA SFT ({train_size} samples, r=32) + DPO Alignment (7B)")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | Base: {base_model_path}")
    print("=" * 82)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================== STAGE 1: SCALED MEDICAL SFT ====================
    print(f"\n[STAGE 1] Loading {train_size} Train / {val_size} Held-Out Val Medical Flashcards...")
    raw_ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    raw_ds = raw_ds.shuffle(seed=42).select(range(train_size + val_size))

    def fmt_sft(ex):
        return {
            "text": (
                "<|im_start|>system\nYou are MedVision Clinical AI, a board-certified medical assistant providing "
                "accurate, concise, evidence-based clinical reasoning.<|im_end|>\n"
                f"<|im_start|>user\n{(ex.get('input') or '').strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{(ex.get('output') or '').strip()}<|im_end|>"
            )
        }

    ds = raw_ds.map(fmt_sft, remove_columns=raw_ds.column_names)
    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, train_size + val_size))

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_logger = MetricsLogger()
    sft_args = SFTConfig(
        output_dir=stage1_dir,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="none",
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        callbacks=[sft_logger],
    )

    t0 = time.perf_counter()
    sft_trainer.train()
    sft_eval = sft_trainer.evaluate()
    sft_elapsed = round(time.perf_counter() - t0, 1)
    sft_loss = round(float(sft_eval["eval_loss"]), 4)
    sft_ppl = round(math.exp(min(sft_loss, 20.0)), 2)

    sft_trainer.save_model(stage1_dir)
    print(f"\n[STAGE 1 COMPLETE] {sft_elapsed}s | Val Loss: {sft_loss} | Val Perplexity: {sft_ppl}")

    # Free Stage 1 trainer memory before Stage 2 DPO
    del sft_trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== STAGE 2: CLINICAL SAFETY DPO ALIGNMENT ====================
    print("\n[STAGE 2] Loading Stage-1 SFT Adapter for Direct Preference Optimization (DPO)...")
    base_model_dpo = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model_dpo = prepare_model_for_kbit_training(base_model_dpo)
    dpo_model = PeftModel.from_pretrained(base_model_dpo, stage1_dir, is_trainable=True)

    dpo_ds = Dataset.from_list(CLINICAL_SAFETY_DPO_PAIRS)
    dpo_logger = MetricsLogger()

    dpo_args = DPOConfig(
        output_dir=stage2_dir,
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="no",
        max_length=512,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=dpo_model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
        callbacks=[dpo_logger],
    )

    t1 = time.perf_counter()
    dpo_res = dpo_trainer.train()
    dpo_elapsed = round(time.perf_counter() - t1, 1)
    dpo_trainer.save_model(stage2_dir)

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    report = {
        "hardware": torch.cuda.get_device_name(0),
        "base_model": base_model_path,
        "stage1_sft": {
            "train_samples": train_size,
            "val_samples": val_size,
            "lora_r": 32,
            "lora_alpha": 64,
            "elapsed_sec": sft_elapsed,
            "val_loss": sft_loss,
            "val_perplexity": sft_ppl,
            "train_logs": sft_logger.train_logs,
            "eval_logs": sft_logger.eval_logs,
        },
        "stage2_dpo": {
            "dpo_pairs": len(CLINICAL_SAFETY_DPO_PAIRS),
            "elapsed_sec": dpo_elapsed,
            "final_dpo_loss": round(float(dpo_res.training_loss), 4),
            "train_logs": dpo_logger.train_logs,
        },
        "peak_vram_gb": peak_vram,
        "final_checkpoint": os.path.abspath(stage2_dir),
    }

    with open("./checkpoints/upgrade1_two_stage_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 82)
    print(f"  UPGRADE 1 COMPLETE! (SFT: {sft_elapsed}s | DPO: {dpo_elapsed}s)")
    print(f"  Stage-1 Scaled SFT Val Perplexity : {sft_ppl} (Loss: {sft_loss})")
    print(f"  Stage-2 Clinical DPO Final Loss   : {dpo_res.training_loss:.4f}")
    print(f"  Peak RTX 4080 VRAM Used           : {peak_vram} GB / 12.0 GB")
    print(f"  Final Aligned 7B Checkpoint       : {os.path.abspath(stage2_dir)}")
    print("=" * 82)


if __name__ == "__main__":
    main()