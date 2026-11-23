import json, os, re, time, torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

THORACIC_ENTITIES = {
    "no_acute":      [r"\bnormal\b", r"\bno acute\b", r"\bno active\b", r"\bclear lungs?\b", r"\bunremarkable\b"],
    "cardiomegaly":  [r"\bcardiomegal", r"\benlarged heart\b", r"\bheart size is enlarged\b", r"\bmildly enlarged cardiac\b"],
    "edema":         [r"\bedema\b", r"\bvascular congestion\b", r"\bfluid overload\b"],
    "consolidation": [r"\bconsolidat", r"\bairspace disease\b", r"\bairspace opacit"],
    "pneumonia":     [r"\bpneumoni", r"\binfectious process\b"],
    "atelectasis":   [r"\batelecta", r"\bvolume loss\b"],
    "pneumothorax":  [r"\bpneumothorax\b"],
    "effusion":      [r"\beffusion\b", r"\bpleural fluid\b"],
    "opacity":       [r"\bopacit", r"\binfiltrat"],
    "nodule_gran":   [r"\bnodule", r"\bgranuloma", r"\bmass\b"],
    "emphysema":     [r"\bemphysema", r"\bcopd\b", r"\bhyperinflat"],
    "degenerative":  [r"\bdegenerative\b", r"\bspondylosis\b", r"\bosteophyte"],
}
NEGATION_CUES = r"(?:\bno\b|\bwithout\b|\bnegative for\b|\bclear of\b|\bfree of\b|\babsence of\b|\bresolved\b|\bnone\b|\bno evidence of\b|\brules? out\b)"


def clean_deid(text: str) -> str:
    t = re.sub(r"\bX{2,}\b", "", (text or ""))
    t = re.sub(r"\s{2,}", " ", t).replace(" .", ".").replace(" ,", ",").strip()
    return t


def extract_clinical_vector(text: str):
    """CheXbert-style Negation-Aware 12-Class Thoracic Pathology Extractor."""
    vec = np.zeros(len(THORACIC_ENTITIES), dtype=np.int32)
    low = (text or "").lower()
    sentences = re.split(r"[.;\n]+", low)
    for idx, (key, patterns) in enumerate(THORACIC_ENTITIES.items()):
        if key == "no_acute":
            if any(re.search(p, low) for p in patterns):
                vec[idx] = 1
            continue
        for sent in sentences:
            for p in patterns:
                m = re.search(p, sent)
                if m:
                    prefix = sent[max(0, m.start() - 48):m.start()]
                    if not re.search(NEGATION_CUES, prefix):
                        vec[idx] = 1
                        break
            if vec[idx] == 1:
                break
    return vec


def rouge_l_f1(pred: str, ref: str) -> float:
    p_toks = re.findall(r"\w+", (pred or "").lower())
    r_toks = re.findall(r"\w+", (ref or "").lower())
    if not p_toks or not r_toks:
        return 0.0
    m, n = len(p_toks), len(r_toks)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if p_toks[i - 1] == r_toks[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    prec, rec = lcs / m, lcs / n
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def evaluate_radiology_reports(model, tok, test_records, label, batch_size=8):
    tok.padding_side = "left"
    prompts = []
    for r in test_records:
        msgs = [
            {"role": "system", "content": "You are a board-certified thoracic radiologist. Synthesize a concise, accurate clinical Impression strictly grounded in the provided Indication and Findings. Do not hallucinate pathologies that are negated."},
            {"role": "user", "content": f"Indication: {r['indication']}\nFindings: {r['findings']}\nWrite the concise Radiology Impression:"}
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    preds_txt = []
    t0 = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inp, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = outs[:, inp["input_ids"].shape[1]:]
        preds_txt.extend([t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)])
        done = min(start + batch_size, len(prompts))
        if done % 50 == 0 or done == len(prompts):
            print(f"  [{label}] Generated {done}/{len(prompts)} test impressions ({time.perf_counter()-t0:.0f}s)", flush=True)

    y_true = np.vstack([extract_clinical_vector(r["impression"]) for r in test_records])
    y_pred = np.vstack([extract_clinical_vector(t) for t in preds_txt])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    micro_prec = tp / max(tp + fp, 1)
    micro_rec  = tp / max(tp + fn, 1)
    micro_f1   = (2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-8)) * 100.0
    neg_spec   = tn / max(tn + fp, 1) * 100.0

    per_f1 = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0:
            tp_j = int(((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum())
            fp_j = int(((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum())
            fn_j = int(((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum())
            denom = 2 * tp_j + fp_j + fn_j
            per_f1.append((2 * tp_j / denom * 100.0) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0
    avg_rouge_l = float(np.mean([rouge_l_f1(p, r["impression"]) for p, r in zip(preds_txt, test_records)])) * 100.0

    return {
        "clinical_micro_f1_pct": round(micro_f1, 2),
        "clinical_macro_f1_pct": round(macro_f1, 2),
        "clinical_precision_pct": round(micro_prec * 100, 2),
        "clinical_recall_pct": round(micro_rec * 100, 2),
        "negation_specificity_pct": round(neg_spec, 2),
        "rouge_l_pct": round(avg_rouge_l, 2),
        "sample_pred": preds_txt[0] if preds_txt else "",
        "sample_ref": test_records[0]["impression"] if test_records else "",
    }


def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"
    base_path = os.environ.get("MEDVISION_BASE_MODEL", "./qwen7b_local")
    prev_adapter = "./checkpoints/medvision-7b-v2-multitask-sft"
    out_adapter = "./checkpoints/medvision-7b-iu-xray-radiologist"

    print("=" * 88)
    print("  PHASE 3B: IU-Xray Thoracic Radiology Report Generation & CheXbert-Style Clinical F1")
    print("=" * 88)

    # Load ykumards/open-i ('train' split only -> deterministic patient/uid-disjoint train/test split)
    raw_ds = load_dataset("ykumards/open-i", split="train").shuffle(seed=42)
    clean_recs = []
    seen_uids = set()
    for ex in raw_ds:
        uid = str(ex.get("uid", ""))
        if uid in seen_uids:
            continue
        fnd = clean_deid(ex.get("findings") or "")
        imp = clean_deid(ex.get("impression") or "")
        ind = clean_deid(ex.get("indication") or "Chest radiograph evaluation")
        if len(fnd) >= 20 and len(imp) >= 8 and fnd.lower() != "none" and imp.lower() != "none":
            seen_uids.add(uid)
            clean_recs.append({"uid": uid, "indication": ind, "findings": fnd, "impression": imp})

    test_recs  = clean_recs[:250]
    train_recs = clean_recs[250:]
    print(f"[IU-XRAY READY] Unique Clean Studies: {len(clean_recs)} -> Train: {len(train_recs)} | Held-Out Test: {len(test_recs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    print("\n>>> [1/3] Evaluating Baseline Base Qwen2.5-7B-Instruct on 250 Held-Out IU-Xray Reports...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb, device_map="auto").eval()
    base_metrics = evaluate_radiology_reports(base_model, tok, test_recs, "Base Qwen2.5-7B")
    print(f"  [BASE] Clinical Micro-F1: {base_metrics['clinical_micro_f1_pct']}% | Negation Specificity: {base_metrics['negation_specificity_pct']}% | ROUGE-L: {base_metrics['rouge_l_pct']}%", flush=True)

    print(f"\n>>> [2/3] Fine-Tuning MedVision-7B-Radiologist on {len(train_recs)} IU-Xray Reports (RTX 4080)...", flush=True)
    tok.padding_side = "right"
    def build_text(r):
        msgs = [
            {"role": "system", "content": "You are a board-certified thoracic radiologist. Synthesize a concise, accurate clinical Impression strictly grounded in the provided Indication and Findings. Do not hallucinate pathologies that are negated."},
            {"role": "user", "content": f"Indication: {r['indication']}\nFindings: {r['findings']}\nWrite the concise Radiology Impression:"},
            {"role": "assistant", "content": r["impression"]}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    tr_ds = Dataset.from_list([build_text(r) for r in train_recs])
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, prev_adapter, is_trainable=True)

    sft_args = SFTConfig(
        output_dir=out_adapter,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=tr_ds, processing_class=tok)
    t0 = time.perf_counter()
    trainer.train()
    train_sec = round(time.perf_counter() - t0, 1)
    trainer.save_model(out_adapter)
    print(f"  [TRAIN COMPLETE in {train_sec}s] Saved adapter to {out_adapter}", flush=True)

    print("\n>>> [3/3] Evaluating Specialized MedVision-7B-Radiologist on 250 Held-Out IU-Xray Reports...", flush=True)
    model.eval()
    rad_metrics = evaluate_radiology_reports(model, tok, test_recs, "MedVision-7B-Radiologist")

    summary = {
        "dataset": f"ykumards/open-i (IU-Xray UID-Disjoint: {len(train_recs)} Train / {len(test_recs)} Test)",
        "train_seconds": train_sec,
        "base_qwen25_7b": base_metrics,
        "medvision_7b_radiologist": rad_metrics,
        "gains": {
            "clinical_micro_f1_delta": round(rad_metrics["clinical_micro_f1_pct"] - base_metrics["clinical_micro_f1_pct"], 2),
            "clinical_macro_f1_delta": round(rad_metrics["clinical_macro_f1_pct"] - base_metrics["clinical_macro_f1_pct"], 2),
            "negation_specificity_delta": round(rad_metrics["negation_specificity_pct"] - base_metrics["negation_specificity_pct"], 2),
            "rouge_l_delta": round(rad_metrics["rouge_l_pct"] - base_metrics["rouge_l_pct"], 2),
        },
        "checkpoint": os.path.abspath(out_adapter),
    }
    with open("./checkpoints/phase3b_iu_xray_clinical_f1.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 88)
    print(f"  IU-XRAY HELD-OUT TEST BENCHMARK (n={len(test_recs)} Real Radiologist Reports)")
    print("=" * 88)
    print(f"  Metric                         | Base Qwen2.5-7B | MedVision-7B-Radiologist |   Gain")
    print("-" * 88)
    print(f"  Clinical Entity Micro-F1       | {base_metrics['clinical_micro_f1_pct']:14.2f}% | {rad_metrics['clinical_micro_f1_pct']:23.2f}% | {summary['gains']['clinical_micro_f1_delta']:+6.2f}%")
    print(f"  Clinical Entity Macro-F1       | {base_metrics['clinical_macro_f1_pct']:14.2f}% | {rad_metrics['clinical_macro_f1_pct']:23.2f}% | {summary['gains']['clinical_macro_f1_delta']:+6.2f}%")
    print(f"  Negation Specificity (No Hal.) | {base_metrics['negation_specificity_pct']:14.2f}% | {rad_metrics['negation_specificity_pct']:23.2f}% | {summary['gains']['negation_specificity_delta']:+6.2f}%")
    print(f"  ROUGE-L Fluency F1             | {base_metrics['rouge_l_pct']:14.2f}% | {rad_metrics['rouge_l_pct']:23.2f}% | {summary['gains']['rouge_l_delta']:+6.2f}%")
    print("=" * 88)

if __name__ == "__main__":
    main()
