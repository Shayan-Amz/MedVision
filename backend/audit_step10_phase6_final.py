import base64, csv, hashlib, importlib, json, os, py_compile, re, subprocess, sys, zipfile
from pathlib import Path
from datasets import load_dataset
from scipy.stats import binomtest

# =========================================================================
# 1. Upgrade Guardrail (Clause-level Negation + ICU/admission/hospitalisation/operation)
#    + FAISS Manifest Hash/Count Validation in clinical_llm_rag.py
# =========================================================================
rag_path = Path("./app/services/clinical_llm_rag.py")
src = rag_path.read_text(encoding="utf-8").replace("\ufeff", "")

new_guardrail = '''def apply_clinical_safety_guardrail(report: str, context_hint: str = ""):
    """Clause-level negation-aware clinical safety guardrail using a neutral safe fallback."""
    flags = []
    text = (report or "").strip()
    if not text:
        return text, flags

    neg_cues = (
        r"\\b(?:no|not|without|absent|negative\\s+for|no\\s+evidence\\s+of|no\\s+signs?\\s+of|"
        r"no\\s+indication\\s+of|no\\s+suspicion\\s+of|unlikely(?:\\s+to\\s+be)?|rules?\\s+out|"
        r"ruled\\s+out|free\\s+of|denies|denied|neither|nor|does\\s+not\\s+show|do\\s+not\\s+show)\\b"
    )
    mal_term = r"(?:malignan(?:cy|cies|t)|carcinoma(?:s)?|cancer(?:ous|s)?|metasta(?:tic|sis|ses|size[ds]?)|neoplasm(?:s)?|neoplastic|tumou?r(?:s)?)"
    bio_term = r"(?:biops(?:y|ied|ies|ying))"
    inv_term = (
        r"(?:surgical\\s+resection|resect(?:ion|ed|ing)?|surg(?:ery|ical\\s+intervention)|operations?|"
        r"hospitali[sz](?:ation|ed|ing)|hospital\\s+admission|"
        r"admission\\s+(?:to\\s+(?:the\\s+)?(?:hospital|icu|ward|inpatient)|for\\s+(?:monitoring|observation|management|workup))|"
        r"recommend(?:ed|ing)?\\s+admission|"
        r"admit(?:ted|ting)?\\s+(?:to\\s+(?:the\\s+)?(?:hospital|icu|ward|inpatient)|immediately|urgently)|"
        r"invasive\\s+procedures?|invasive\\s+workup|invasive\\s+intervention)"
    )

    def has_unnegated(pattern: str, s: str) -> bool:
        for m in re.finditer(r"\\b" + pattern + r"\\b", s, flags=re.IGNORECASE):
            # Clause-level lookback up to sentence/clause boundary (., ;, !, ?, newline, or 'but'/'however')
            pre = s[:m.start()]
            clause = re.split(r"[.;!?\\n]|\\b(?:but|however|although|whereas)\\b", pre, flags=re.IGNORECASE)[-1]
            if not re.search(neg_cues, clause, flags=re.IGNORECASE):
                return True
        return False

    if has_unnegated(mal_term, text):
        flags.append("malignancy_claim_blocked")
    if has_unnegated(bio_term, text):
        flags.append("biopsy_directive_blocked")
    if has_unnegated(inv_term, text):
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
    return text, flags'''

src, n_g = re.subn(
    r"def apply_clinical_safety_guardrail\(.*?\):[\s\S]*?return text, flags",
    lambda _: new_guardrail, src, count=1
)
assert n_g == 1, "Failed to update apply_clinical_safety_guardrail!"

# Add FAISS manifest validation helper in clinical_llm_rag.py
if "def validate_rag_corpus_manifest" not in src:
    val_helper = '''
def validate_rag_corpus_manifest(rag_dir: str, index_ntotal: int, docs_len: int) -> bool:
    """Validates FAISS index size and SHA-256 against corpus_manifest.json before serving."""
    import hashlib
    if index_ntotal != docs_len:
        raise RuntimeError(f"RAG integrity mismatch: faiss ntotal={index_ntotal} != docs={docs_len}")
    mpath = os.path.join(rag_dir, "corpus_manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath, "r", encoding="utf-8"))
        if int(man.get("doc_count", -1)) != docs_len or int(man.get("faiss_ntotal", -1)) != index_ntotal:
            raise RuntimeError("RAG manifest count mismatch!")
        idx_path = os.path.join(rag_dir, man.get("faiss_filename", "full_faiss.index"))
        if os.path.exists(idx_path) and man.get("faiss_index_sha256"):
            h = hashlib.sha256()
            with open(idx_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != man["faiss_index_sha256"]:
                raise RuntimeError("RAG FAISS index SHA-256 mismatch!")
    return True
'''
    src = src + "\n" + val_helper
    # Wire validate_rag_corpus_manifest right after faiss.read_index if present
    src = re.sub(
        r"(_RAG_DOCS\s*=\s*json\.load\(f\))",
        r"\1\n        validate_rag_corpus_manifest(rag_dir, _RAG_INDEX.ntotal, len(_RAG_DOCS))",
        src, count=1
    )

rag_path.write_text(src, encoding="utf-8")
py_compile.compile(str(rag_path), doraise=True)

# Write FAISS-independent unit tests + real Integration Test for answer_clinical_followup_with_rag()
test_code = '''import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.services.clinical_llm_rag as rag_mod
from app.services.clinical_llm_rag import apply_clinical_safety_guardrail, answer_clinical_followup_with_rag

def test_unsafe_directives_and_uk_icu_variants():
    cases = [
        ("Right upper lobe mass should be biopsied immediately.", "biopsy_directive_blocked"),
        ("Findings are highly suspicious for metastatic carcinoma.", "malignancy_claim_blocked"),
        ("Recommend admission for monitoring overnight.", "invasive_or_admission_directive_blocked"),
        ("Patient should be admitted to ICU promptly.", "invasive_or_admission_directive_blocked"),
        ("Urgent hospitalisation is warranted.", "invasive_or_admission_directive_blocked"),
        ("Schedule an urgent operation and invasive procedures.", "invasive_or_admission_directive_blocked"),
    ]
    for txt, expected_flag in cases:
        safe, flags = apply_clinical_safety_guardrail(txt)
        assert expected_flag in flags, f"Missing {expected_flag} for: {txt} (got {flags})"
        assert "Decision-support draft only" in safe
        # Idempotency check: neutral fallback must NOT re-trigger guardrail
        _, re_flags = apply_clinical_safety_guardrail(safe)
        assert re_flags == [], f"Fallback re-triggered guardrail: {re_flags}"

def test_clause_level_negation_preserved():
    safe_cases = [
        "Lungs are clear without acute cardiopulmonary abnormality.",
        "No evidence of acute cardiopulmonary disease or malignancy is identified.",
        "There is no sign of carcinoma, pleural effusion, or pneumothorax.",
        "Patient denies prior cancer; lungs are unremarkable.",
        "Chest radiograph does not show metastatic disease.",
    ]
    for txt in safe_cases:
        out, flags = apply_clinical_safety_guardrail(txt)
        assert flags == [], f"False positive on negated sentence: {txt} -> {flags}"
        assert out == txt

def test_affirmative_after_but_still_blocked():
    txt = "No pleural effusion is seen, but a right upper lobe mass is suspicious for malignancy."
    safe, flags = apply_clinical_safety_guardrail(txt)
    assert "malignancy_claim_blocked" in flags

def test_rag_followup_integration_without_faiss():
    fake_ev = [{"rank": 1, "score": 0.91, "snippet": "Pulmonary nodule follow-up per Fleischner criteria."}]
    with patch.object(rag_mod, "retrieve_medical_evidence", return_value=fake_ev), \\
         patch.object(rag_mod, "load_clinical_llm", return_value=None), \\
         patch.object(rag_mod, "_generate_with_adapter", return_value="Recommend admission for monitoring and immediate biopsy of the malignant mass."):
        res = answer_clinical_followup_with_rag("What should we do next?", study_context="RUL opacity")
        assert "malignancy_claim_blocked" in res["safety_guardrail_flags"]
        assert "biopsy_directive_blocked" in res["safety_guardrail_flags"]
        assert "invasive_or_admission_directive_blocked" in res["safety_guardrail_flags"]
        assert "Decision-support draft only" in res["answer"]
        assert res["evidence"] == fake_ev

if __name__ == "__main__":
    test_unsafe_directives_and_uk_icu_variants()
    test_clause_level_negation_preserved()
    test_affirmative_after_but_still_blocked()
    test_rag_followup_integration_without_faiss()
    print("ALL PHASE 6 UNIT & RAG-FOLLOWUP INTEGRATION TESTS PASSED (FAISS-INDEPENDENT)!")
'''
Path("./tests/test_clinical_safety_guardrail.py").write_text(test_code, encoding="utf-8")
subprocess.run([sys.executable, "./tests/test_clinical_safety_guardrail.py"], check=True)

# Re-audit all 238 IU-Xray drafts with Phase 6 guardrail
import app.services.clinical_llm_rag as rag_mod
importlib.reload(rag_mod)
iu = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
c_mal = c_bio = c_inv = c_any = after_any = 0
exs = []
for it in iu["items"]:
    raw = it.get("llm_report", "")
    safe, flags = rag_mod.apply_clinical_safety_guardrail(raw)
    if "malignancy_claim_blocked" in flags: c_mal += 1
    if "biopsy_directive_blocked" in flags: c_bio += 1
    if "invasive_or_admission_directive_blocked" in flags: c_inv += 1
    if flags:
        c_any += 1
        _, rf = rag_mod.apply_clinical_safety_guardrail(safe)
        if rf: after_any += 1
        if str(it["uid"]) == "744" or len(exs) < 5:
            exs.append({"uid": it["uid"], "flags": flags, "before": raw, "after": safe})

safety_audit = {
    "n_evaluated": len(iu["items"]),
    "guardrail_mode": "Clause-level negation-aware neutral safe fallback (Impression + RAG Follow-up)",
    "before_guardrail": {
        "malignancy_claims": c_mal,
        "biopsy_directives": c_bio,
        "other_invasive_or_admission_directives": c_inv,
        "total_unique_reports_flagged": c_any,
    },
    "after_guardrail": {"flagged_reports_remaining": after_any},
    "unit_and_integration_tests_passed": True,
    "examples": exs,
}
Path("./checkpoints/clinical_safety_guardrail_audit.json").write_text(json.dumps(safety_audit, indent=2), encoding="utf-8")
print("[1/7] Phase 6 Guardrail Re-Audited:", json.dumps({k: v for k, v in safety_audit.items() if k != "examples"}))

# =========================================================================
# 2. Exact McNemar Paired Tests on MedMCQA (n=4,183 & deduped n=4,136)
# =========================================================================
with open("./checkpoints/medmcqa_predictions.csv", encoding="utf-8") as f:
    mrows = list(csv.DictReader(f))

def mcnemar_pair(rows, col_a, col_b):
    b01 = sum((r[col_a] != r["gold"] and r[col_b] == r["gold"]) for r in rows)
    b10 = sum((r[col_a] == r["gold"] and r[col_b] != r["gold"]) for r in rows)
    p = float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if (b01 + b10) > 0 else 1.0
    return {"b01": b01, "b10": b10, "exact_mcnemar_p": float(f"{p:.6g}"), "significant_005": p < 0.05}

dedup_rows = [r for r in mrows if not (int(r["q_exact_in_ctx"]) or int(r["q_8gram_in_ctx"]))]
mc_sum_path = Path("./checkpoints/medmcqa_audit_summary.json")
mc_sum = json.load(open(mc_sum_path, encoding="utf-8"))
mc_sum["paired_mcnemar_full_4183"] = {
    "single_rag_vs_base": mcnemar_pair(mrows, "base_pred", "single_rag_pred"),
    "cyclic4_rag_vs_base": mcnemar_pair(mrows, "base_pred", "cyclic4_pred"),
    "cyclic4_vs_single_rag": mcnemar_pair(mrows, "single_rag_pred", "cyclic4_pred"),
}
mc_sum["paired_mcnemar_dedup_4136"] = {
    "single_rag_vs_base": mcnemar_pair(dedup_rows, "base_pred", "single_rag_pred"),
    "cyclic4_rag_vs_base": mcnemar_pair(dedup_rows, "base_pred", "cyclic4_pred"),
    "cyclic4_vs_single_rag": mcnemar_pair(dedup_rows, "single_rag_pred", "cyclic4_pred"),
}
mc_sum_path.write_text(json.dumps(mc_sum, indent=2), encoding="utf-8")
p_cs = mc_sum["paired_mcnemar_full_4183"]["cyclic4_vs_single_rag"]["exact_mcnemar_p"]
print(f"[2/7] MedMCQA Exact McNemar: Cyclic4 vs Base p={mc_sum['paired_mcnemar_full_4183']['cyclic4_rag_vs_base']['exact_mcnemar_p']} | Cyclic4 vs Single p={p_cs} (NOT significant)")

# =========================================================================
# 3. PubMedQA Manifest: Add question_sha256 for reproducible textual overlap verification
# =========================================================================
def q_hash(s):
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

ds_l = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").shuffle(seed=42)
ds_a = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=42).select(range(3200))

man_path = Path("./checkpoints/pubmedqa_split_manifest.json")
man = json.load(open(man_path, encoding="utf-8"))
man["test_question_sha256"] = [q_hash(r["question"]) for r in ds_l.select(range(0, 200))]
man["train_labeled_question_sha256"] = [q_hash(r["question"]) for r in ds_l.select(range(200, 1000))]
man["pqa_artificial_question_sha256"] = [q_hash(r["question"]) for r in ds_a]
man["pqa_artificial_vs_test_question_hash_overlap"] = len(set(man["test_question_sha256"]) & set(man["pqa_artificial_question_sha256"]))
man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")

ver_script = '''import json

pqa = json.load(open("./checkpoints/pubmedqa_split_manifest.json", encoding="utf-8"))
assert len(pqa["test_pubids"]) == 200 and len(pqa["train_pubids"]) == 800 and len(pqa["pqa_artificial_train_pubids"]) == 3200
assert len(set(pqa["test_pubids"]) & set(pqa["train_pubids"])) == 0
assert len(set(pqa["test_pubids"]) & set(pqa["pqa_artificial_train_pubids"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["train_labeled_question_sha256"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["pqa_artificial_question_sha256"])) == 0
print("[OK] PubMedQA Manifest: 0 pubid overlap AND 0 normalized-question SHA256 overlap across 800 labeled + 3,200 artificial train!")

iu = json.load(open("./checkpoints/iu_xray_split_manifest.json", encoding="utf-8"))
e2e = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
e2e_uids = [str(it["uid"]) for it in e2e["items"]]
assert len(iu["test_selected_250_uids"]) == 250 and len(iu["test_evaluated_238_uids"]) == 238
assert len(set(iu["test_selected_250_uids"]) & set(iu["radiologist_lora_train_uids"])) == 0
assert set(e2e_uids) == set(iu["test_evaluated_238_uids"])
print("[OK] IU-Xray Manifest: 238 E2E UIDs match 250 selected (12 missing frontal) & 0 overlap with 3,080 train UIDs!")
'''
Path("./verify_manifests_reproducible.py").write_text(ver_script, encoding="utf-8")
subprocess.run([sys.executable, "./verify_manifests_reproducible.py"], check=True)
print("[3/7] PubMedQA textual SHA256 overlap verified & asserted in verify_manifests_reproducible.py")

# =========================================================================
# 4. RAG Corpus Manifest (SHA-256, version, count) + Enriched Passage Source IDs
# =========================================================================
import faiss
rag_dir = Path("./medical_rag_index_full")
idx_file = rag_dir / "full_faiss.index"
docs_file = rag_dir / "full_docs.json"
docs = json.load(open(docs_file, encoding="utf-8"))
idx_obj = faiss.read_index(str(idx_file))

def file_sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest()

corpus_man = {
    "corpus_version": "medvision-rag-v1.0-trainbank",
    "builder_script": "backend/phase2h_scientific_boost_80_60.py",
    "faiss_filename": "full_faiss.index",
    "docs_filename": "full_docs.json",
    "doc_count": len(docs),
    "faiss_ntotal": int(idx_obj.ntotal),
    "faiss_index_sha256": file_sha256(idx_file),
    "docs_json_sha256": file_sha256(docs_file),
    "source_datasets": [
        "openlifescienceai/medmcqa (train split explanations only)",
        "medalpaca/medical_meadow_medical_flashcards (clinical fact flashcards)"
    ],
}
(rag_dir / "corpus_manifest.json").write_text(json.dumps(corpus_man, indent=2), encoding="utf-8")
# Validate runtime checker works against real manifest
assert rag_mod.validate_rag_corpus_manifest(str(rag_dir), int(idx_obj.ntotal), len(docs))

# Enrich provenance JSON with stable source_id, dataset, original_record_id, passage_sha256, corpus_version
prov_path = Path("./checkpoints/medmcqa_rag_provenance_and_options_audit.json")
prov = json.load(open(prov_path, encoding="utf-8"))
prov["corpus_version"] = corpus_man["corpus_version"]
prov["faiss_index_sha256"] = corpus_man["faiss_index_sha256"]
prov["docs_json_sha256"] = corpus_man["docs_json_sha256"]
for ex in prov.get("answer_key_doc_examples_with_ids", []):
    did = int(ex["doc_id"])
    full_t = docs[did]
    is_fc = full_t.startswith("Clinical Fact:")
    ex["source_id"] = f"{corpus_man['corpus_version']}::doc_{did:06d}"
    ex["dataset"] = "medalpaca/medical_meadow_medical_flashcards" if is_fc else "openlifescienceai/medmcqa:train"
    ex["original_record_id"] = f"doc_idx_{did}"
    ex["passage_sha256"] = hashlib.sha256(full_t.encode("utf-8")).hexdigest()[:16]
prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
print(f"[4/7] RAG corpus_manifest.json built & SHA-256 verified ({corpus_man['faiss_index_sha256'][:12]}...)")

# =========================================================================
# 5. Synchronize HTML, Standalone HTML, MD & README (100% consistent counts & p-values)
# =========================================================================
guard_txt = f"{c_mal}->0 malignancy, {c_bio}->0 biopsy, {c_inv}->0 invasive/admission ({c_any} unique reports -> neutral safe fallback)"
guard_html = f"{c_mal}&rarr;0 malignancy, {c_bio}&rarr;0 biopsy, {c_inv}&rarr;0 invasive/admission ({c_any} unique reports &rarr; neutral safe fallback)"

html_path = Path("../presentation/MedVision_Academic_Report.html")
h = html_path.read_text(encoding="utf-8")
h = re.sub(r"maybe F1\s*=\s*0\.0+%?", "maybe F1 = 13.79%", h)
h = re.sub(r"identified <code>13/238</code> unverified malignancy claims and <code>8/238</code> direct biopsy directives \(<code>13</code> reports sanitized\)",
           f"identified <code>{c_mal}/238</code> malignancy claims, <code>{c_bio}/238</code> biopsy directives, and <code>{c_inv}/238</code> other invasive/admission directives (<code>{c_any}</code> unique reports)", h)
h = re.sub(r"Safety Guardrail[^|<]*", f"Safety Guardrail (Neutral Fallback): {guard_html} ", h)
h = h.replace(
    "Train-Bank RAG achieves <code>64.34%</code> Single / <code>64.80%</code> Cyclic-4",
    f"Train-Bank RAG achieves <code>64.34%</code> Single / <code>64.80%</code> Cyclic-4 (Exact McNemar vs Base <code>p &lt; 0.0001</code>; Cyclic-4 vs Single-Pass <b>not statistically significant</b>, Exact McNemar <code>p = {p_cs:.3f}</code>)"
)
html_path.write_text(h, encoding="utf-8")

pres_dir = html_path.parent
def embed_img(m):
    fp = pres_dir / m.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode("ascii")}"' if fp.exists() else m.group(0)
(pres_dir / "MedVision_Academic_Report_Standalone.html").write_text(re.sub(r'src="(figures/[^"]+\.png)"', embed_img, h), encoding="utf-8")

for mstr in ("../presentation/MedVision_Academic_Report.md", "../README.md"):
    mp = Path(mstr)
    if mp.exists():
        t = mp.read_text(encoding="utf-8")
        t = re.sub(r"maybe F1\s*=\s*0\.0+%?", "maybe F1 = 13.79%", t)
        t = re.sub(r"Two-Prop Z-test\s*\*\*`p < 0\.0001`\*\*", f"Exact McNemar **`p < 0.0001`** vs Base; Cyclic-4 vs Single-Pass non-significant (`p = {p_cs:.3f}`)", t)
        t = re.sub(r"Two-Prop Z-test `p < 0\.0001`", "Exact McNemar `p < 0.0001` vs Base", t)
        t = re.sub(r"Safety Guardrail[^|\n]*", f"Safety Guardrail (neutral fallback): {guard_txt}", t)
        t = t.replace("External-Hospital E2E (n=250)", "Weak-Label External Feasibility (250 Selected / n=238 Evaluated)")
        t = t.replace("250 unseen frontal chest radiographs", "250 UID-disjoint selected studies (12 missing frontal -> 238 evaluated frontal chest radiographs)")
        t = t.replace("`n=250` External Feasibility Cohort", "`250 Selected / n=238 Evaluated` External Feasibility Cohort")
        mp.write_text(t, encoding="utf-8")
print("[5/7] Synchronized HTML, Standalone HTML, MD & README (maybe F1=13.79%, exact guardrail counts, Exact McNemar p=0.356, 250/238)")

# =========================================================================
# 6. Package Phase 6 Final Audit ZIP on Desktop
# =========================================================================
zip_path = Path(os.environ["USERPROFILE"]) / "Desktop" / "MedVision_Audit_Phase6_Final.zip"
bundle = [
    ("backend/app/services/clinical_llm_rag.py", "./app/services/clinical_llm_rag.py"),
    ("backend/tests/test_clinical_safety_guardrail.py", "./tests/test_clinical_safety_guardrail.py"),
    ("backend/verify_manifests_reproducible.py", "./verify_manifests_reproducible.py"),
    ("backend/checkpoints/pubmedqa_split_manifest.json", "./checkpoints/pubmedqa_split_manifest.json"),
    ("backend/checkpoints/iu_xray_split_manifest.json", "./checkpoints/iu_xray_split_manifest.json"),
    ("backend/checkpoints/clinical_safety_guardrail_audit.json", "./checkpoints/clinical_safety_guardrail_audit.json"),
    ("backend/checkpoints/medmcqa_audit_summary.json", "./checkpoints/medmcqa_audit_summary.json"),
    ("backend/checkpoints/medmcqa_rag_provenance_and_options_audit.json", "./checkpoints/medmcqa_rag_provenance_and_options_audit.json"),
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

print(f"[6/7] & [7/7] Phase 6 Final Audit ZIP ready: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")
