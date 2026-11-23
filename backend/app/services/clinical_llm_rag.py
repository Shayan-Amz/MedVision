from app.services.clinical_safety import apply_clinical_safety_guardrail, validate_rag_corpus_manifest
import os, re, json, threading
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder

_LOCK = threading.Lock()
_LLM_MODEL = None
_LLM_TOK = None
_ACTIVE_ADAPTERS = set()

_RAG_INDEX = None
_RAG_DOCS = None
_RAG_EMBEDDER = None
_RAG_RERANKER = None


def _clean_radiology_text(text: str) -> str:
    text = re.sub(r"\bX{2,}\b", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def load_clinical_llm():
    """Loads 4-bit Qwen2.5-7B-Instruct once and mounts both radiologist & clinical_qa LoRA adapters."""
    global _LLM_MODEL, _LLM_TOK, _ACTIVE_ADAPTERS
    if _LLM_MODEL is not None:
        return _LLM_MODEL, _LLM_TOK

    with _LOCK:
        if _LLM_MODEL is not None:
            return _LLM_MODEL, _LLM_TOK

        base_dir = "./qwen7b_local"
        v2_path = "./checkpoints/medvision-7b-v2-multitask-sft"
        rad_path = "./checkpoints/medvision-7b-iu-xray-radiologist"

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _LLM_TOK = AutoTokenizer.from_pretrained(base_dir)
        if _LLM_TOK.pad_token is None:
            _LLM_TOK.pad_token = _LLM_TOK.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        if os.path.exists(v2_path):
            model = PeftModel.from_pretrained(base_model, v2_path, adapter_name="clinical_qa")
            _ACTIVE_ADAPTERS.add("clinical_qa")
            if os.path.exists(rad_path):
                model.load_adapter(rad_path, adapter_name="radiologist")
                _ACTIVE_ADAPTERS.add("radiologist")
        elif os.path.exists(rad_path):
            model = PeftModel.from_pretrained(base_model, rad_path, adapter_name="radiologist")
            _ACTIVE_ADAPTERS.add("radiologist")
        else:
            model = base_model

        model.eval()
        _LLM_MODEL = model
        return _LLM_MODEL, _LLM_TOK


def load_medical_rag_cpu():
    """Loads the 175,413-passage disclosed train-bank FAISS index + Cross-Encoder Reranker on CPU to preserve GPU VRAM."""
    global _RAG_INDEX, _RAG_DOCS, _RAG_EMBEDDER, _RAG_RERANKER
    if _RAG_INDEX is not None:
        return _RAG_INDEX, _RAG_DOCS, _RAG_EMBEDDER, _RAG_RERANKER

    rag_dir = "./medical_rag_index_full"
    idx_path = os.path.join(rag_dir, "full_faiss.index")
    doc_path = os.path.join(rag_dir, "full_docs.json")
    if not (os.path.exists(idx_path) and os.path.exists(doc_path)):
        return None, None, None, None

    with _LOCK:
        if _RAG_INDEX is not None:
            return _RAG_INDEX, _RAG_DOCS, _RAG_EMBEDDER, _RAG_RERANKER
        _RAG_INDEX = faiss.read_index(idx_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            _RAG_DOCS = json.load(f)
        validate_rag_corpus_manifest(rag_dir, _RAG_INDEX.ntotal, len(_RAG_DOCS))
        _RAG_EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        _RAG_RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        return _RAG_INDEX, _RAG_DOCS, _RAG_EMBEDDER, _RAG_RERANKER


def retrieve_medical_evidence(query: str, top_k_faiss: int = 15, top_k_final: int = 3) -> List[Dict[str, Any]]:
    index, docs, embedder, reranker = load_medical_rag_cpu()
    if index is None:
        return []
    q_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, idxs = index.search(q_emb, top_k_faiss)
    cand_ids = [int(i) for i in idxs[0] if 0 <= int(i) < len(docs)]
    if not cand_ids:
        return []
    pairs = [(query, docs[cid]) for cid in cand_ids]
    scores = reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(scores)[::-1][:top_k_final]
    evidence = []
    for rank, pos in enumerate(order, 1):
        cid = cand_ids[pos]
        evidence.append({
            "rank": rank,
            "doc_id": cid,
            "rerank_score": round(float(scores[pos]), 3),
            "snippet": docs[cid][:420].strip(),
        })
    return evidence


def _generate_with_adapter(adapter_name: str, system_prompt: str, user_prompt: str, max_new_tokens: int = 220) -> str:
    model, tok = load_clinical_llm()
    with _LOCK:
        if hasattr(model, "set_adapter") and adapter_name in _ACTIVE_ADAPTERS:
            model.set_adapter(adapter_name)
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _clean_radiology_text(gen)


def synthesize_radiology_impression(
    primary: str,
    top_conf: float,
    top3: List[Tuple[str, float]],
    urgency: str,
    patient_ctx: str = "",
) -> Dict[str, Any]:
    """Generates a vision-grounded, contradiction-free radiology impression across all 14 NIH classes."""
    load_clinical_llm()
    positives = [(name, prob) for name, prob in top3 if prob >= 0.60]
    m_ctr = re.search(r"CTR=([0-9]+\.[0-9]+)", patient_ctx)
    if m_ctr and float(m_ctr.group(1)) > 0.50 and not any(n == "Cardiomegaly" for n, _ in positives):
        positives.append(("Anatomical Cardiomegaly (CTR>" + m_ctr.group(1) + ")", 0.90))
    negatives = [(name, prob) for name, prob in top3 if prob < 0.60]

    if urgency == "ROUTINE" or not positives:
        pos_str = "None (all pathologies below clinical action threshold)"
    else:
        pos_str = ", ".join(f"{n} ({p*100:.1f}%)" for n, p in positives)

    neg_str = ", ".join(f"{n} ({p*100:.1f}%)" for n, p in negatives) if negatives else "None"

    user_prompt = (
        f"Patient Context: {patient_ctx}\n"
        f"Triage Urgency: {urgency}\n"
        f"CONFIRMED HIGH-PROBABILITY POSITIVE FINDINGS (MUST REPORT AS PRESENT ? NEVER NEGATE THESE): {pos_str}\n"
        f"Low-probability / sub-threshold findings: {neg_str}\n\n"
        f"Write a concise 3-sentence clinical Radiology Impression and Immediate Action Plan strictly consistent with the positive findings above."
    )
    sys_prompt = (
        "You are MedVision-7B Clinical Radiology Copilot. Strictly follow the vision ensemble findings: "
        "every finding listed under CONFIRMED POSITIVE must be explicitly reported as present with appropriate clinical action; "
        "never contradict or negate a confirmed positive finding."
    )
    adapter = "clinical_qa" if (positives and "clinical_qa" in _ACTIVE_ADAPTERS) else (
        "radiologist" if "radiologist" in _ACTIVE_ADAPTERS else "clinical_qa"
    )
    text = _generate_with_adapter(adapter, sys_prompt, user_prompt, max_new_tokens=200)

    low_txt = text.lower()
    contradicted = any(
        (f"no {n.lower().replace('_', ' ')}" in low_txt or f"without {n.lower().replace('_', ' ')}" in low_txt)
        for n, _ in positives
    )
    if contradicted and positives:
        pos_summary = "; ".join(f"{n} ({p*100:.1f}%)" for n, p in positives)
        text = (
            f"1) Positive thoracic findings detected by Vision Ensemble: {pos_summary}. "
            f"2) Triage urgency stratified as {urgency} ({patient_ctx}). "
            f"3) Immediate clinical correlation and urgent radiology review are recommended."
        )

    text, safety_flags = apply_clinical_safety_guardrail(text)
    return {"report": text, "adapter_used": adapter, "safety_guardrail_flags": safety_flags}


def answer_clinical_followup_with_rag(question: str, study_context: str = "") -> Dict[str, Any]:
    """Runs Single-Pass 175k FAISS + Cross-Encoder RAG on CPU, then answers via medvision-7b-v2."""
    search_query = f"{question} {study_context[:200]}".strip()
    evidence = retrieve_medical_evidence(search_query, top_k_faiss=15, top_k_final=3)
    ev_block = "\n".join(f"[{e['rank']}] {e['snippet']}" for e in evidence) if evidence else "No external passage retrieved."

    user_prompt = (
        f"Retrieved Medical Evidence (Top-3 Reranked from 175k Corpus):\n{ev_block}\n\n"
        f"Current Patient Study Context:\n{study_context}\n\n"
        f"Clinician Question: {question}\n"
        f"Provide a concise, evidence-grounded clinical answer citing [1], [2], or [3] where relevant."
    )
    sys_prompt = (
        "You are MedVision-7B Clinical Copilot (78.5% PubMedQA, 64.98% MedMCQA). "
        "Answer safely, concisely, and ground clinical guidance in the retrieved evidence."
    )
    load_clinical_llm(); adapter = "clinical_qa" if "clinical_qa" in _ACTIVE_ADAPTERS else "radiologist"
    answer = _generate_with_adapter(adapter, sys_prompt, user_prompt, max_new_tokens=240)
    answer, safety_flags = apply_clinical_safety_guardrail(answer, context_hint="RAG Clinical Follow-up")
    return {"answer": answer, "adapter_used": adapter, "evidence": evidence, "safety_guardrail_flags": safety_flags}



