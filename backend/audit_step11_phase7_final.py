import base64, hashlib, json, os, py_compile, re, subprocess, sys, zipfile
from pathlib import Path
from datasets import load_dataset

# =========================================================================
# 1. Create Standalone Zero-Dependency Module: app/services/clinical_safety.py
# =========================================================================
safety_mod_code = '''"""Standalone, zero-ML-dependency clinical safety guardrail and RAG corpus integrity validator."""
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

_PRE_NEG_CUES = re.compile(
    r"\\b(?:no|not|without|absent|negative\\s+for|no\\s+evidence\\s+of|no\\s+signs?\\s+of|"
    r"no\\s+indication\\s+of|no\\s+suspicion\\s+of|unlikely(?:\\s+to\\s+be)?|rules?\\s+out|"
    r"ruled\\s+out|free\\s+of|denies|denied|neither|nor|does\\s+not\\s+show|do\\s+not\\s+show)\\b",
    re.IGNORECASE,
)
_POST_NEG_CUES = re.compile(
    r"^\\s*(?:is|are|was|were)?\\s*(?:not\\s+(?:seen|identified|detected|present|found|evident|"
    r"demonstrated|confirmed|suspected|indicated|recommended|warranted|required)|"
    r"absent|excluded|ruled\\s+out|unlikely)\\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.;!?\\n]|\\b(?:but|however|although|whereas|yet|and)\\b", re.IGNORECASE)

_MAL_TERM = r"(?:malignan(?:cy|cies|t)|carcinoma(?:s)?|cancer(?:ous|s)?|metasta(?:tic|sis|ses|size[ds]?)|neoplasm(?:s)?|neoplastic|tumou?r(?:s)?)"
_BIO_TERM = r"(?:biops(?:y|ied|ies|ying))"
_INV_TERM = (
    r"(?:surgical\\s+resection|resect(?:ion|ed|ing)?|surg(?:ery|ical\\s+intervention)|operations?|"
    r"hospitali[sz](?:ation|ed|ing)|hospital\\s+admission|"
    r"admission\\s+(?:to\\s+(?:the\\s+)?(?:hospital|icu|ward|inpatient)|for\\s+(?:monitoring|observation|management|workup))|"
    r"recommend(?:ed|ing)?\\s+admission|"
    r"admit(?:ted|ting)?\\s+(?:to\\s+(?:the\\s+)?(?:hospital|icu|ward|inpatient)|immediately|urgently)|"
    r"invasive\\s+procedures?|invasive\\s+workup|invasive\\s+intervention)"
)


def _has_unnegated(pattern: str, text: str) -> bool:
    for m in re.finditer(r"\\b" + pattern + r"\\b", text, flags=re.IGNORECASE):
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
            "Automated imaging review detected an indeterminate finding requiring clinician correlation."
            + ctx_part
            + " Definitive diagnostic labeling and direct management directives were withheld by the safety filter. "
            "Prompt attending radiologist and physician evaluation is recommended. "
            "[Safety Filter: Decision-support draft only; clinician verification required.]"
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
'''
Path("./app/services/clinical_safety.py").write_text(safety_mod_code, encoding="utf-8")
py_compile.compile("./app/services/clinical_safety.py", doraise=True)

# Wire clinical_llm_rag.py to import from standalone clinical_safety.py
rag_path = Path("./app/services/clinical_llm_rag.py")
src = rag_path.read_text(encoding="utf-8").replace("\ufeff", "")
src = re.sub(r"def apply_clinical_safety_guardrail\(.*?\):[\s\S]*?return text, flags\n*", "", src, count=1)
src = re.sub(r"def validate_rag_corpus_manifest\([\s\S]*?return True\n*", "", src, count=1)
if "from app.services.clinical_safety import" not in src:
    src = "from app.services.clinical_safety import apply_clinical_safety_guardrail, validate_rag_corpus_manifest\n" + src
rag_path.write_text(src, encoding="utf-8")
py_compile.compile(str(rag_path), doraise=True)

# =========================================================================
# 2. Update RAG corpus_manifest.json with encoder_model & reranker_model
# =========================================================================
rag_dir = Path("./medical_rag_index_full")
man_rag_path = rag_dir / "corpus_manifest.json"
man_rag = json.load(open(man_rag_path, encoding="utf-8"))
man_rag["encoder_model"] = "BAAI/bge-small-en-v1.5"
man_rag["reranker_model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
man_rag_path.write_text(json.dumps(man_rag, indent=2), encoding="utf-8")

# Validate real manifest with BOTH SHA-256 checks
from app.services.clinical_safety import apply_clinical_safety_guardrail, validate_rag_corpus_manifest
validate_rag_corpus_manifest(str(rag_dir), man_rag["faiss_ntotal"], man_rag["doc_count"], expected_encoder="BAAI/bge-small-en-v1.5")
print("[1/6] & [2/6] Standalone clinical_safety.py created & BOTH FAISS + docs SHA-256 validated!")

# =========================================================================
# 3. FAISS-Free Pytest Unit & Integration Tests (tests/test_clinical_safety_guardrail.py)
# =========================================================================
test_code = '''import hashlib, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.clinical_safety import apply_clinical_safety_guardrail, validate_rag_corpus_manifest

def test_mixed_negation_and_affirmative_biopsy_flagged():
    txt = "No evidence of malignancy and biopsy is recommended."
    safe, flags = apply_clinical_safety_guardrail(txt)
    assert "biopsy_directive_blocked" in flags
    assert "malignancy_claim_blocked" not in flags
    assert "Decision-support draft only" in safe

def test_post_negated_cancer_not_flagged():
    for txt in (
        "Cancer is not identified.",
        "Malignancy is excluded on current exam.",
        "Biopsy is not indicated at this time.",
        "No evidence of acute cardiopulmonary disease or malignancy.",
        "Patient denies prior cancer; lungs are unremarkable.",
    ):
        out, flags = apply_clinical_safety_guardrail(txt)
        assert flags == [], f"False positive on negated sentence: {txt} -> {flags}"
        assert out == txt

def test_unsafe_directives_and_variants_blocked_idempotently():
    cases = [
        ("Right upper lobe mass should be biopsied immediately.", "biopsy_directive_blocked"),
        ("Findings are highly suspicious for metastatic carcinoma.", "malignancy_claim_blocked"),
        ("Recommend admission for monitoring overnight.", "invasive_or_admission_directive_blocked"),
        ("Patient should be admitted to ICU promptly.", "invasive_or_admission_directive_blocked"),
        ("Urgent hospitalisation is warranted.", "invasive_or_admission_directive_blocked"),
        ("Schedule an urgent operation and invasive procedures.", "invasive_or_admission_directive_blocked"),
    ]
    for txt, expected in cases:
        safe, flags = apply_clinical_safety_guardrail(txt)
        assert expected in flags, f"Expected {expected} for {txt}, got {flags}"
        _, re_flags = apply_clinical_safety_guardrail(safe)
        assert re_flags == [], f"Fallback re-triggered: {re_flags}"

def test_rag_corpus_manifest_validates_both_index_and_docs_sha256():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        idx_b, doc_b = b"fake-faiss-bytes", b'["doc1","doc2"]'
        (p / "idx.faiss").write_bytes(idx_b)
        (p / "docs.json").write_bytes(doc_b)
        man = {
            "corpus_version": "test-v1",
            "faiss_filename": "idx.faiss",
            "docs_filename": "docs.json",
            "encoder_model": "BAAI/bge-small-en-v1.5",
            "doc_count": 2,
            "faiss_ntotal": 2,
            "faiss_index_sha256": hashlib.sha256(idx_b).hexdigest(),
            "docs_json_sha256": hashlib.sha256(doc_b).hexdigest(),
        }
        (p / "corpus_manifest.json").write_text(json.dumps(man), encoding="utf-8")
        assert validate_rag_corpus_manifest(td, 2, 2, expected_encoder="BAAI/bge-small-en-v1.5")["corpus_version"] == "test-v1"

        # Tamper with docs.json -> MUST raise RuntimeError
        (p / "docs.json").write_bytes(b'["tampered","doc2"]')
        try:
            validate_rag_corpus_manifest(td, 2, 2)
            assert False, "Expected SHA-256 mismatch error on docs.json!"
        except RuntimeError as e:
            assert "docs JSON SHA-256" in str(e)

def test_rag_followup_wrapper_contract_without_faiss():
    # Verify clinical_llm_rag.py wires guardrail into both synthesis & RAG follow-up without importing FAISS
    src = Path(__file__).resolve().parent.parent.joinpath("app/services/clinical_llm_rag.py").read_text(encoding="utf-8")
    assert "from app.services.clinical_safety import apply_clinical_safety_guardrail, validate_rag_corpus_manifest" in src
    assert src.count("apply_clinical_safety_guardrail(") >= 2
    # Mocked follow-up flow applying the exact same guardrail function
    mock_raw = "No evidence of malignancy and biopsy is recommended; admit to ICU."
    safe, flags = apply_clinical_safety_guardrail(mock_raw, context_hint="RAG Clinical Follow-up")
    assert "biopsy_directive_blocked" in flags and "invasive_or_admission_directive_blocked" in flags
    assert "Decision-support draft only" in safe

if __name__ == "__main__":
    test_mixed_negation_and_affirmative_biopsy_flagged()
    test_post_negated_cancer_not_flagged()
    test_unsafe_directives_and_variants_blocked_idempotently()
    test_rag_corpus_manifest_validates_both_index_and_docs_sha256()
    test_rag_followup_wrapper_contract_without_faiss()
    print("ALL PHASE 7 FAISS-FREE UNIT & INTEGRATION TESTS PASSED!")
'''
Path("./tests/test_clinical_safety_guardrail.py").write_text(test_code, encoding="utf-8")
subprocess.run([sys.executable, "./tests/test_clinical_safety_guardrail.py"], check=True)
# Also run via pytest if installed
r_pt = subprocess.run([sys.executable, "-m", "pytest", "-q", "./tests/test_clinical_safety_guardrail.py"], capture_output=True, text=True)
print("[3/6] Pytest output:", (r_pt.stdout + r_pt.stderr).strip())

# Re-audit 238 IU-Xray drafts
iu = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
c_mal = c_bio = c_inv = c_any = after_any = 0
for it in iu["items"]:
    safe, flags = apply_clinical_safety_guardrail(it.get("llm_report", ""))
    if "malignancy_claim_blocked" in flags: c_mal += 1
    if "biopsy_directive_blocked" in flags: c_bio += 1
    if "invasive_or_admission_directive_blocked" in flags: c_inv += 1
    if flags:
        c_any += 1
        if apply_clinical_safety_guardrail(safe)[1]: after_any += 1

sa = {
    "n_evaluated": len(iu["items"]),
    "guardrail_module": "app/services/clinical_safety.py (zero ML/FAISS dependencies)",
    "before_guardrail": {"malignancy_claims": c_mal, "biopsy_directives": c_bio, "other_invasive_or_admission_directives": c_inv, "total_unique_reports_flagged": c_any},
    "after_guardrail": {"flagged_reports_remaining": after_any},
}
Path("./checkpoints/clinical_safety_guardrail_audit.json").write_text(json.dumps(sa, indent=2), encoding="utf-8")

# =========================================================================
# 4. Full 64-char SHA-256 hashes in pubmedqa_split_manifest.json + verify script
# =========================================================================
def full_sha256(s):
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

ds_l = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
ds_a = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=42).select(range(3200))
man_path = Path("./checkpoints/pubmedqa_split_manifest.json")
man = json.load(open(man_path, encoding="utf-8"))
man["hash_algorithm"] = "SHA-256 (full 64-hex-char on lowercase alphanumeric normalized question)"
man["test_question_sha256"] = [full_sha256(r["question"]) for r in ds_l.select(range(0, 200))]
man["train_labeled_question_sha256"] = [full_sha256(r["question"]) for r in ds_l.select(range(200, 1000))]
man["pqa_artificial_question_sha256"] = [full_sha256(r["question"]) for r in ds_a]
man["pqa_artificial_vs_test_question_hash_overlap"] = len(set(man["test_question_sha256"]) & set(man["pqa_artificial_question_sha256"]))
man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

ver_code = '''import json

pqa = json.load(open("./checkpoints/pubmedqa_split_manifest.json", encoding="utf-8"))
assert len(pqa["test_pubids"]) == 200 and len(pqa["train_pubids"]) == 800 and len(pqa["pqa_artificial_train_pubids"]) == 3200
assert all(len(h) == 64 for h in pqa["test_question_sha256"])
assert all(len(h) == 64 for h in pqa["pqa_artificial_question_sha256"])
assert len(set(pqa["test_pubids"]) & set(pqa["train_pubids"])) == 0
assert len(set(pqa["test_pubids"]) & set(pqa["pqa_artificial_train_pubids"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["train_labeled_question_sha256"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["pqa_artificial_question_sha256"])) == 0
print("[OK] PubMedQA Manifest: 0 pubid & 0 full 64-char SHA-256 question overlap (200 test vs 800 labeled + 3,200 artificial)")

iu = json.load(open("./checkpoints/iu_xray_split_manifest.json", encoding="utf-8"))
e2e = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
e2e_uids = [str(it["uid"]) for it in e2e["items"]]
assert len(iu["test_selected_250_uids"]) == 250 and len(iu["test_evaluated_238_uids"]) == 238
assert len(set(iu["test_selected_250_uids"]) & set(iu["radiologist_lora_train_uids"])) == 0
assert set(e2e_uids) == set(iu["test_evaluated_238_uids"])
print("[OK] IU-Xray Manifest: 238 E2E UIDs match 250 selected (12 missing frontal) & 0 overlap with 3,080 train UIDs")
'''
Path("./verify_manifests_reproducible.py").write_text(ver_code, encoding="utf-8")
subprocess.run([sys.executable, "./verify_manifests_reproducible.py"], check=True)
print("[4/6] Full 64-char SHA-256 PubMedQA manifest & IU-Xray manifest verified!")

# =========================================================================
# 5. Scrub ALL remaining stale stats across HTML, Standalone HTML, MD & README
# =========================================================================
def final_doc_scrub(t: str) -> str:
    # Remove uncited Human Comparable claim
    t = re.sub(r";?\s*Comparable to [Hh]uman(?: avg)?\s*\([^)]*\)", "", t)
    t = re.sub(r"\|\s*Comparable to [Hh]uman\s*\([^)]*\)", "", t)
    # Fix Zero-Overlap RAG -> Disclosed Train-Bank RAG
    t = re.sub(r"Zero[- ]Overlap\s+RAG", "Disclosed Train-Bank RAG", t, flags=re.IGNORECASE)
    t = re.sub(r"Audited Zero-Overlap / Disclosed Train-Bank", "Disclosed Train-Bank", t)
    t = re.sub(r"Disclosed Train-Bank / Zero-Overlap Audited", "Disclosed Train-Bank", t)
    # Fix old RAG dedup counts (6 / 16 -> 40 exact / 13 8-gram / 47 union)
    t = t.replace("0.143% exact / 0.383% 8-gram", "40 exact / 13 8-gram / 47 union removed (n=4,136)")
    t = t.replace("6/4,183", "40/4,183").replace("16/4,183", "13/4,183 (47 union)")
    # Fix old IU-Xray 104/146 & confusion counts -> 79 abnormal / 159 normal in n=238
    t = t.replace("104 abnormal / 146 normal", "79 abnormal / 159 normal (n=238 evaluated from 250 selected)")
    t = t.replace("`104` abnormal / `146` normal", "`79` abnormal / `159` normal (`n=238` evaluated from `250` selected)")
    t = t.replace("86/104", "71/79").replace("53/146", "65/159")
    t = t.replace("73/104", "59/79").replace("101/146", "110/159")
    t = t.replace("55/104", "43/79").replace("132/146", "129/159").replace("55/69", "43/73")
    t = re.sub(r"maybe F1\s*=\s*0\.0+%?", "maybe F1 = 13.79%", t)
    return t

html_path = Path("../presentation/MedVision_Academic_Report.html")
h = final_doc_scrub(html_path.read_text(encoding="utf-8"))
html_path.write_text(h, encoding="utf-8")
pres_dir = html_path.parent
def embed_img(m):
    fp = pres_dir / m.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode("ascii")}"' if fp.exists() else m.group(0)
(pres_dir / "MedVision_Academic_Report_Standalone.html").write_text(re.sub(r'src="(figures/[^"]+\.png)"', embed_img, h), encoding="utf-8")

cm_note = (
    "\n- **IU-Xray Confusion Matrices (`n=238` evaluated from `250` selected; `79` abnormal / `159` normal):** "
    "ER Triage (`TP=71, FN=8, TN=65, FP=94`; `89.9% Sens / 40.9% Spec`); "
    "Balanced (`TP=59, FN=20, TN=110, FP=49`; `74.7% Sens / 69.2% Spec`); "
    "Alert Gate (`TP=43, FN=36, TN=129, FP=30`; `54.4% Sens / 81.1% Spec`). "
    "MedMCQA Disclosed Train-Bank RAG deduplication: `40` exact question matches, `13` 8-gram matches, `47` union removed (`n=4,136` remaining: `64.34%` Single / `64.80%` Cyclic-4; Cyclic-4 vs Single non-significant, Exact McNemar `p=0.356`).\n"
)
for mstr in ("../presentation/MedVision_Academic_Report.md", "../README.md"):
    mp = Path(mstr)
    if mp.exists():
        t = final_doc_scrub(mp.read_text(encoding="utf-8"))
        if "TP=71, FN=8, TN=65, FP=94" not in t:
            t = t.rstrip() + cm_note
        mp.write_text(t, encoding="utf-8")
print("[5/6] Scrubbed & synced HTML, Standalone HTML, MD & README (79/159, 71/8/65/94, 40/13/47, maybe F1=13.79%, no uncited Human claim)")

# =========================================================================
# 6. Package Phase 7 Final Audit ZIP on Desktop
# =========================================================================
zip_path = Path(os.environ["USERPROFILE"]) / "Desktop" / "MedVision_Audit_Phase7_Final.zip"
bundle = [
    ("backend/app/services/clinical_safety.py", "./app/services/clinical_safety.py"),
    ("backend/app/services/clinical_llm_rag.py", "./app/services/clinical_llm_rag.py"),
    ("backend/tests/test_clinical_safety_guardrail.py", "./tests/test_clinical_safety_guardrail.py"),
    ("backend/verify_manifests_reproducible.py", "./verify_manifests_reproducible.py"),
    ("backend/checkpoints/pubmedqa_split_manifest.json", "./checkpoints/pubmedqa_split_manifest.json"),
    ("backend/checkpoints/iu_xray_split_manifest.json", "./checkpoints/iu_xray_split_manifest.json"),
    ("backend/checkpoints/iu_xray_e2e_audit.json", "./checkpoints/iu_xray_e2e_audit.json"),
    ("backend/checkpoints/clinical_safety_guardrail_audit.json", "./checkpoints/clinical_safety_guardrail_audit.json"),
    ("backend/checkpoints/medmcqa_audit_summary.json", "./checkpoints/medmcqa_audit_summary.json"),
    ("backend/medical_rag_index_full/corpus_manifest.json", "./medical_rag_index_full/corpus_manifest.json"),
    ("presentation/MedVision_Academic_Report.html", "../presentation/MedVision_Academic_Report.html"),
    ("presentation/MedVision_Academic_Report.md", "../presentation/MedVision_Academic_Report.md"),
    ("README.md", "../README.md"),
]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for arcname, srcp in bundle:
        p = Path(srcp)
        if p.exists():
            zf.write(p, arcname=arcname)
        else:
            print("WARNING missing:", srcp)

print(f"[6/6] Phase 7 Final Audit ZIP ready: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")

