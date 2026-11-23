import hashlib, json, sys, tempfile
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
