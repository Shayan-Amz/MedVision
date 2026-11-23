"""Standalone, zero-ML-dependency clinical safety guardrail and RAG corpus integrity validator."""
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

_PRE_NEG_CUES = re.compile(
    r"\b(?:no|not|without|absent|negative\s+for|no\s+evidence\s+of|no\s+signs?\s+of|"
    r"no\s+indication\s+of|no\s+suspicion\s+of|unlikely(?:\s+to\s+be)?|rules?\s+out|"
    r"ruled\s+out|free\s+of|denies|denied|neither|nor|does\s+not\s+show|do\s+not\s+show)\b",
    re.IGNORECASE,
)
_POST_NEG_CUES = re.compile(
    r"^\s*(?:is|are|was|were)?\s*(?:not\s+(?:seen|identified|detected|present|found|evident|"
    r"demonstrated|confirmed|suspected|indicated|recommended|warranted|required)|"
    r"absent|excluded|ruled\s+out|unlikely)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.;!?\n]|\b(?:but|however|although|whereas|yet|and)\b", re.IGNORECASE)

_MAL_TERM = r"(?:malignan(?:cy|cies|t)|carcinoma(?:s)?|cancer(?:ous|s)?|metasta(?:tic|sis|ses|size[ds]?)|neoplasm(?:s)?|neoplastic|tumou?r(?:s)?)"
_BIO_TERM = r"(?:biops(?:y|ied|ies|ying))"
_INV_TERM = (
    r"(?:surgical\s+resection|resect(?:ion|ed|ing)?|surg(?:ery|ical\s+intervention)|operations?|"
    r"hospitali[sz](?:ation|ed|ing)|hospital\s+admission|"
    r"admission\s+(?:to\s+(?:the\s+)?(?:hospital|icu|ward|inpatient)|for\s+(?:monitoring|observation|management|workup))|"
    r"recommend(?:ed|ing)?\s+admission|"
    r"admit(?:ted|ting)?\s+(?:to\s+(?:the\s+)?(?:hospital|icu|ward|inpatient)|immediately|urgently)|"
    r"invasive\s+procedures?|invasive\s+workup|invasive\s+intervention)"
)


def _has_unnegated(pattern: str, text: str) -> bool:
    for m in re.finditer(r"\b" + pattern + r"\b", text, flags=re.IGNORECASE):
        pre = text[:m.start()]
        post = text[m.end():m.end() + 80]
        local_pre = _CLAUSE_BREAK.split(pre)[-1]
        if _PRE_NEG_CUES.search(local_pre):
            continue
        if _POST_NEG_CUES.search(post):
            continue
        return True
    return False


def apply_clinical_safety_guardrail(report: str, context_hint: str = "") -> Tuple[str, List[str]]:
    """Pre- and post-negation-aware clinical safety guardrail using a neutral safe fallback."""
    flags: List[str] = []
    text = (report or "").strip()
    if not text:
        return text, flags

    if _has_unnegated(_MAL_TERM, text):
        flags.append("malignancy_claim_blocked")
    if _has_unnegated(_BIO_TERM, text):
        flags.append("biopsy_directive_blocked")
    if _has_unnegated(_INV_TERM, text):
        flags.append("invasive_or_admission_directive_blocked")

    if flags:
        ctx_part = f" Reference: {context_hint}." if context_hint else ""
        safe_fallback = (
            "Automated text suppression was applied by the Lexical Clinical Safety Filter."
            + ctx_part
            + " Direct diagnostic labeling and management directives were withheld. "
            "Qualified clinician and radiologist review is required to interpret this study. "
            "[Lexical Clinical Safety Filter: Decision-support draft only; clinician verification required.]"
        )
        return safe_fallback, flags
    return text, flags


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_rag_corpus_manifest(
    rag_dir: str,
    index_ntotal: int,
    docs_len: int,
    expected_encoder: str = None,
) -> Dict[str, object]:
    """Validates FAISS index count, docs count, encoder version, and SHA-256 of BOTH index and docs JSON."""
    if index_ntotal != docs_len:
        raise RuntimeError(f"RAG integrity mismatch: faiss ntotal={index_ntotal} != docs={docs_len}")
    mpath = os.path.join(rag_dir, "corpus_manifest.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing corpus_manifest.json in {rag_dir}")
    with open(mpath, "r", encoding="utf-8") as f:
        man = json.load(f)

    if int(man.get("doc_count", -1)) != docs_len or int(man.get("faiss_ntotal", -1)) != index_ntotal:
        raise RuntimeError("RAG manifest doc_count / faiss_ntotal mismatch!")
    if expected_encoder and man.get("encoder_model") != expected_encoder:
        raise RuntimeError(f"RAG encoder mismatch: {man.get('encoder_model')} != {expected_encoder}")

    idx_path = os.path.join(rag_dir, man["faiss_filename"])
    docs_path = os.path.join(rag_dir, man["docs_filename"])
    if _sha256_file(idx_path) != man["faiss_index_sha256"]:
        raise RuntimeError("RAG FAISS index SHA-256 mismatch!")
    if _sha256_file(docs_path) != man["docs_json_sha256"]:
        raise RuntimeError("RAG docs JSON SHA-256 mismatch!")
    return man
