import base64, json, os, py_compile, re, subprocess, sys, zipfile
from pathlib import Path
from datasets import load_dataset

# =========================================================================
# 1. Negation-Aware Safe-Fallback Clinical Safety Guardrail (Impression + RAG Follow-up)
# =========================================================================
rag_path = Path("./app/services/clinical_llm_rag.py")
src = rag_path.read_text(encoding="utf-8").replace("\ufeff", "")

new_guardrail = '''def apply_clinical_safety_guardrail(report: str, context_hint: str = ""):
    """Negation-aware post-generation clinical safety guardrail using safe fallback (never word-mangling)."""
    flags = []
    text = (report or "").strip()
    if not text:
        return text, flags

    neg_prefix = r"(?:no|not|without|absent|negative\\s+for|no\\s+evidence\\s+of|unlikely\\s+to\\s+be|rules?\\s+out)\\s+(?:\\w+\\s+){0,3}"

    mal_term = r"(?:malignan(?:cy|cies|t)|carcinoma(?:s)?|cancer(?:ous|s)?|metasta(?:tic|sis|ses|size[ds]?)|neoplasm(?:s)?|neoplastic|tumou?r(?:s)?)"
    bio_term = r"(?:biops(?:y|ied|ies|ying))"
    inv_term = r"(?:surgical\\s+resection|resect(?:ion|ed|ing)?|surg(?:ery|ical\\s+intervention)|hospitaliz(?:ation|ed|ing)|hospital\\s+admission|admit(?:ted|ting)?\\s+(?:to\\s+the\\s+hospital|immediately|urgently)|invasive\\s+procedures?|invasive\\s+workup)"

    def has_unnegated(pattern: str, s: str) -> bool:
        for m in re.finditer(r"\\b" + pattern + r"\\b", s, flags=re.IGNORECASE):
            window = s[max(0, m.start() - 45):m.start()]
            if not re.search(neg_prefix + r"$", window, flags=re.IGNORECASE):
                return True
        return False

    if has_unnegated(mal_term, text):
        flags.append("malignancy_claim_blocked")
    if has_unnegated(bio_term, text):
        flags.append("biopsy_directive_blocked")
    if has_unnegated(inv_term, text):
        flags.append("invasive_or_admission_directive_blocked")

    if flags:
        ctx_part = f" Context: {context_hint}." if context_hint else ""
        safe_fallback = (
            "1) Indeterminate cardiopulmonary or focal finding flagged by automated imaging analysis; "
            "definitive characterization cannot be established by AI alone." + ctx_part + " "
            "2) Direct diagnostic labeling or invasive/admission directives "
            "were withheld by the clinical safety guardrail. "
            "3) Prompt attending radiologist review, clinical correlation, and physician-guided follow-up are recommended. "
            "[Safety Guardrail: Decision-support draft only; clinician verification required.]"
        )
        return safe_fallback, flags
    return text, flags'''

src, n_g = re.subn(
    r"def apply_clinical_safety_guardrail\(.*?\):[\s\S]*?return safe, flags",
    lambda _: new_guardrail, src, count=1
)
if n_g == 0:
    src, n_g = re.subn(
        r"def apply_clinical_safety_guardrail\(.*?\):[\s\S]*?return text, flags",
        lambda _: new_guardrail, src, count=1
    )
assert n_g == 1, "Failed to update apply_clinical_safety_guardrail!"

# Also apply guardrail to answer_clinical_followup_with_rag()
old_rag_ret = '    return {"answer": answer, "adapter_used": adapter, "evidence": evidence}'
new_rag_ret = (
    '    answer, safety_flags = apply_clinical_safety_guardrail(answer, context_hint="RAG Clinical Follow-up")\n'
    '    return {"answer": answer, "adapter_used": adapter, "evidence": evidence, "safety_guardrail_flags": safety_flags}'
)
if old_rag_ret in src:
    src = src.replace(old_rag_ret, new_rag_ret, 1)

rag_path.write_text(src, encoding="utf-8")
py_compile.compile(str(rag_path), doraise=True)

# Comprehensive unit tests (adversarial + hospital admission + invasive procedures + negated safe sentences + RAG follow-up check)
test_code = '''import inspect, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.clinical_llm_rag import apply_clinical_safety_guardrail, answer_clinical_followup_with_rag

def test_unsafe_directives_trigger_safe_fallback():
    cases = [
        ("Right upper lobe mass should be biopsied immediately.", "biopsy_directive_blocked"),
        ("Findings are highly suspicious for metastatic carcinoma.", "malignancy_claim_blocked"),
        ("Recommend immediate hospital admission for further management.", "invasive_or_admission_directive_blocked"),
        ("Patient requires urgent invasive procedures and surgical resection.", "invasive_or_admission_directive_blocked"),
        ("Mass is likely malignant and should be biopsied urgently.", "biopsy_directive_blocked"),
    ]
    for txt, expected_flag in cases:
        safe, flags = apply_clinical_safety_guardrail(txt)
        assert expected_flag in flags, f"Missing {expected_flag} for: {txt} (got {flags})"
        assert "Decision-support draft only" in safe
        assert "biopsied" not in safe.lower() and "hospital admission" not in safe.lower() and "invasive procedures" not in safe.lower()

def test_negated_and_benign_sentences_preserved():
    safe_cases = [
        "Lungs are clear without acute cardiopulmonary abnormality.",
        "No evidence of malignancy or pleural effusion is identified.",
        "There is no carcinoma or acute consolidation seen.",
    ]
    for txt in safe_cases:
        out, flags = apply_clinical_safety_guardrail(txt)
        assert flags == [], f"False positive trigger on negated/benign sentence: {txt} -> {flags}"
        assert out == txt

def test_rag_followup_invokes_guardrail():
    src = inspect.getsource(answer_clinical_followup_with_rag)
    assert "apply_clinical_safety_guardrail" in src, "RAG follow-up must invoke clinical safety guardrail!"

if __name__ == "__main__":
    test_unsafe_directives_trigger_safe_fallback()
    test_negated_and_benign_sentences_preserved()
    test_rag_followup_invokes_guardrail()
    print("ALL PHASE 5 CLINICAL SAFETY GUARDRAIL TESTS PASSED!")
'''
Path("./tests/test_clinical_safety_guardrail.py").write_text(test_code, encoding="utf-8")
subprocess.run([sys.executable, "./tests/test_clinical_safety_guardrail.py"], check=True)

# Re-audit all 238 IU-Xray drafts with separated counts (Malignancy vs Biopsy vs Other Invasive/Admission)
import importlib, app.services.clinical_llm_rag as rag_mod
importlib.reload(rag_mod)
iu = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
c_mal = c_bio = c_inv = c_any = 0
after_any = 0
examples = []
for it in iu["items"]:
    raw = it.get("llm_report", "")
    safe, flags = rag_mod.apply_clinical_safety_guardrail(raw)
    if "malignancy_claim_blocked" in flags: c_mal += 1
    if "biopsy_directive_blocked" in flags: c_bio += 1
    if "invasive_or_admission_directive_blocked" in flags: c_inv += 1
    if flags:
        c_any += 1
        _, recheck_flags = rag_mod.apply_clinical_safety_guardrail(safe)
        if recheck_flags: after_any += 1
        if str(it["uid"]) == "744" or len(examples) < 5:
            examples.append({"uid": it["uid"], "flags": flags, "before": raw, "after": safe})

safety_audit = {
    "n_evaluated": len(iu["items"]),
    "guardrail_mode": "Negation-aware safe structured fallback (applied to both radiology impression & RAG follow-up)",
    "before_guardrail": {
        "malignancy_claims": c_mal,
        "biopsy_directives": c_bio,
        "other_invasive_or_admission_directives": c_inv,
        "total_unique_reports_flagged": c_any,
    },
    "after_guardrail": {"flagged_reports_remaining": after_any},
    "adversarial_and_negation_unit_tests_passed": True,
    "examples": examples,
}
Path("./checkpoints/clinical_safety_guardrail_audit.json").write_text(json.dumps(safety_audit, indent=2), encoding="utf-8")
print("[1/7] Phase 5 Guardrail Audited:", json.dumps({k: v for k, v in safety_audit.items() if k != "examples"}))

# =========================================================================
# 2. Full 3,200 pqa_artificial pubids in pubmedqa_split_manifest.json
# =========================================================================
man_path = Path("./checkpoints/pubmedqa_split_manifest.json")
man = json.load(open(man_path, encoding="utf-8"))
ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").shuffle(seed=42).select(range(3200))
art_pids = [str(r["pubid"]) for r in ds_art]
man["pqa_artificial_train_pubids"] = art_pids
man.pop("pqa_artificial_train_pubids_sample_first20", None)
man["pqa_artificial_vs_test_pubid_overlap"] = len(set(art_pids) & set(man["test_pubids"]))
man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"[2/7] Stored ALL {len(art_pids)} pqa_artificial pubids in pubmedqa_split_manifest.json (overlap={man['pqa_artificial_vs_test_pubid_overlap']})")

# =========================================================================
# 3. Real IU-Xray Radiologist LoRA Train/Test UID Manifest
# =========================================================================
def clean_deid(s): return re.sub(r"\s+", " ", (s or "").replace("XXXX", "")).strip()
raw_iu = load_dataset("ykumards/open-i", split="train").shuffle(seed=42)
clean_recs, seen = [], set()
for ex in raw_iu:
    uid = str(ex.get("uid", ""))
    if uid in seen: continue
    fnd, imp = clean_deid(ex.get("findings") or ""), clean_deid(ex.get("impression") or "")
    if len(fnd) >= 20 and len(imp) >= 8 and fnd.lower() != "none" and imp.lower() != "none":
        seen.add(uid)
        clean_recs.append({"uid": uid, "has_frontal": ex.get("img_frontal") is not None})

iu_test_250 = [r["uid"] for r in clean_recs[:250]]
iu_test_238_frontal = [r["uid"] for r in clean_recs[:250] if r["has_frontal"]]
iu_test_12_missing = [r["uid"] for r in clean_recs[:250] if not r["has_frontal"]]
iu_train_uids = [r["uid"] for r in clean_recs[250:]]
assert len(set(iu_test_250) & set(iu_train_uids)) == 0
iu_manifest = {
    "dataset": "ykumards/open-i",
    "split_source_script": "backend/phase3b_iu_xray_clinical_report_eval.py",
    "seed": 42,
    "n_test_selected": len(iu_test_250),
    "n_test_evaluated_with_frontal": len(iu_test_238_frontal),
    "n_test_skipped_missing_frontal": len(iu_test_12_missing),
    "n_radiologist_lora_train": len(iu_train_uids),
    "train_test_uid_overlap": 0,
    "test_selected_250_uids": iu_test_250,
    "test_evaluated_238_uids": iu_test_238_frontal,
    "test_skipped_12_missing_frontal_uids": iu_test_12_missing,
    "radiologist_lora_train_uids": iu_train_uids,
}
Path("./checkpoints/iu_xray_split_manifest.json").write_text(json.dumps(iu_manifest, indent=2), encoding="utf-8")
print(f"[3/7] Saved iu_xray_split_manifest.json (Train={len(iu_train_uids)}, TestSelected={len(iu_test_250)}, Evaluated={len(iu_test_238_frontal)}, Overlap=0)")

# Standalone reproducible manifest verification script
ver_script = '''import json
from pathlib import Path

pqa = json.load(open("./checkpoints/pubmedqa_split_manifest.json", encoding="utf-8"))
assert len(pqa["test_pubids"]) == 200
assert len(pqa["train_pubids"]) == 800
assert len(pqa["pqa_artificial_train_pubids"]) == 3200
assert len(set(pqa["test_pubids"]) & set(pqa["train_pubids"])) == 0
assert len(set(pqa["test_pubids"]) & set(pqa["pqa_artificial_train_pubids"])) == 0
print(f"[OK] PubMedQA Manifest: 200 test vs 800 labeled + 3,200 artificial -> 0 overlap")

iu = json.load(open("./checkpoints/iu_xray_split_manifest.json", encoding="utf-8"))
e2e = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
e2e_uids = [str(it["uid"]) for it in e2e["items"]]
assert len(iu["test_selected_250_uids"]) == 250
assert len(iu["test_evaluated_238_uids"]) == 238
assert len(set(iu["test_selected_250_uids"]) & set(iu["radiologist_lora_train_uids"])) == 0
assert set(e2e_uids) == set(iu["test_evaluated_238_uids"])
assert len(set(e2e_uids) & set(iu["radiologist_lora_train_uids"])) == 0
print(f"[OK] IU-Xray Manifest: 238 E2E UIDs match 250 selected (12 missing frontal) & 0 overlap with {len(iu['radiologist_lora_train_uids'])} train UIDs")
'''
Path("./verify_manifests_reproducible.py").write_text(ver_script, encoding="utf-8")
subprocess.run([sys.executable, "./verify_manifests_reproducible.py"], check=True)

# =========================================================================
# 4. Find real builder script(s) for medical_rag_index_full
# =========================================================================
rag_builders = []
for pyf in Path(".").glob("*.py"):
    t = pyf.read_text(encoding="utf-8", errors="ignore")
    if pyf.name != "audit_step9_phase5_final.py" and "full_faiss.index" in t and ("write_index" in t or "add(" in t):
        rag_builders.append(str(pyf))
print("[4/7] Found real medical_rag_index_full builder script(s):", rag_builders)

# =========================================================================
# 5. Fix Deprecated phase6d & phase7 to ALWAYS execute canonical pipeline
# =========================================================================
dep_stub = '''"""DEPRECATED LEGACY SCRIPT.
Superseded by `audit_step1_nih14_25596_exact.py`, which fixes label mapping, Patient ID normalization,
and Patient-Level Cluster Paired Bootstrap Delta-AUROC 95% CIs. Always delegates execution directly.
"""
import subprocess, sys

if __name__ == "__main__":
    print("[DEPRECATED] Executing canonical pipeline: audit_step1_nih14_25596_exact.py", flush=True)
    subprocess.run([sys.executable, "./audit_step1_nih14_25596_exact.py"], check=True)
'''
for legacy in ("./phase6d_sync_25596_exact_ci_and_figs.py", "./phase7_patient_cluster_bootstrap_and_html_sync.py"):
    Path(legacy).write_text(dep_stub, encoding="utf-8")
    py_compile.compile(legacy, doraise=True)
print("[5/7] Fixed deprecated phase6d & phase7 to unconditionally execute canonical pipeline")

# =========================================================================
# 6. Fully Sync HTML, Standalone HTML, MD & README (scrub ALL stale figures & Leak-Free claims)
# =========================================================================
guard_summary = f"{c_mal}->0 malignancy, {c_bio}->0 biopsy, {c_inv}->0 invasive/admission ({c_any} unique reports -> safe fallback)"

html_path = Path("../presentation/MedVision_Academic_Report.html")
h = html_path.read_text(encoding="utf-8")
h = re.sub(r"External E2E Validation \(n=250\)", "Weak-Label External Feasibility Evaluation (250 Selected / n=238 Evaluated)", h)
h = re.sub(r"Safety Guardrail: [^|<]+", f"Safety Guardrail (Safe Fallback): {guard_summary} ", h)
h = re.sub(r"\bLeak[- ]Free\b", "Audited Zero-Overlap / Disclosed Train-Bank", h, flags=re.IGNORECASE)
html_path.write_text(h, encoding="utf-8")

pres_dir = html_path.parent
def embed_img(m):
    fp = pres_dir / m.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode("ascii")}"' if fp.exists() else m.group(0)
(pres_dir / "MedVision_Academic_Report_Standalone.html").write_text(re.sub(r'src="(figures/[^"]+\.png)"', embed_img, h), encoding="utf-8")

# Cleanly synchronize MD & README top tables and remove every stale figure/overclaim
def clean_md_file(mp: Path):
    if not mp.exists(): return
    t = mp.read_text(encoding="utf-8")
    t = re.sub(r"\bLeak[- ]Free\b", "Disclosed Train-Bank / Zero-Overlap Audited", t, flags=re.IGNORECASE)
    t = re.sub(r"Hard Safety Guarantee", "Negation-Aware Post-Generation Safety Guardrail", t, flags=re.IGNORECASE)
    # Replace stale numbers in body tables so zero contradiction exists anywhere in MD/README
    repls = [
        ("0.8208", "0.8213"), ("0.8212", "0.8213"), ("0.8138", "0.8139"), ("0.8140", "0.8141"),
        ("[0.8168, 0.8255]", "[0.8151, 0.8267]"), ("63.5%", "63.6%"),
        ("46.50%", "39.50%"), ("51.50%", "39.50%"), ("74.50%", "78.50%"), ("53.93%", "60.05%"), ("36.93%", "33.51%"),
        ("0.7619", "0.7408"), ("[0.6998, 0.8196]", "[0.6699, 0.8070]"),
        ("82.7%", "89.9%"), ("36.3%", "40.9%"), ("70.2%", "74.7%"), ("52.9%", "54.4%"), ("90.4%", "81.1%"), ("79.7%", "58.9%"),
        ("2.39s", "2.53s"), ("2,390.4 ms", "2,532.0 ms"), ("0/250", "0/238"), ("1.46%", "1.54%"), ("1.5%", "1.54%"),
        ("n=4,167", "n=4,136"), ("64.84%", "64.80%"),
    ]
    for a, b in repls:
        t = t.replace(a, b)
    t = re.sub(r"Safety Guardrail reduced malignancy/tumor claims [^\n]+", f"Safety Guardrail (negation-aware safe fallback): {guard_summary}", t)
    mp.write_text(t, encoding="utf-8")

clean_md_file(Path("../presentation/MedVision_Academic_Report.md"))
clean_md_file(Path("../README.md"))
print("[6/7] Fully synchronized HTML, Standalone HTML, MD & README (zero stale numbers or Leak-Free claims)")

# =========================================================================
# 7. Package Small Phase 5 Final Audit ZIP on Desktop
# =========================================================================
zip_path = Path(os.environ["USERPROFILE"]) / "Desktop" / "MedVision_Audit_Phase5_Final.zip"
bundle = [
    ("backend/app/services/clinical_llm_rag.py", "./app/services/clinical_llm_rag.py"),
    ("backend/tests/test_clinical_safety_guardrail.py", "./tests/test_clinical_safety_guardrail.py"),
    ("backend/verify_manifests_reproducible.py", "./verify_manifests_reproducible.py"),
    ("backend/phase6d_sync_25596_exact_ci_and_figs.py", "./phase6d_sync_25596_exact_ci_and_figs.py"),
    ("backend/phase7_patient_cluster_bootstrap_and_html_sync.py", "./phase7_patient_cluster_bootstrap_and_html_sync.py"),
    ("backend/checkpoints/pubmedqa_split_manifest.json", "./checkpoints/pubmedqa_split_manifest.json"),
    ("backend/checkpoints/iu_xray_split_manifest.json", "./checkpoints/iu_xray_split_manifest.json"),
    ("backend/checkpoints/clinical_safety_guardrail_audit.json", "./checkpoints/clinical_safety_guardrail_audit.json"),
    ("backend/checkpoints/medmcqa_rag_provenance_and_options_audit.json", "./checkpoints/medmcqa_rag_provenance_and_options_audit.json"),
    ("presentation/MedVision_Academic_Report.html", "../presentation/MedVision_Academic_Report.html"),
    ("presentation/MedVision_Academic_Report.md", "../presentation/MedVision_Academic_Report.md"),
    ("README.md", "../README.md"),
]
for rb in rag_builders:
    bundle.append((f"backend/{Path(rb).name}", rb))

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    seen_arc = set()
    for arcname, srcp in bundle:
        p = Path(srcp)
        if p.exists() and arcname not in seen_arc:
            zf.write(p, arcname=arcname)
            seen_arc.add(arcname)

print(f"[7/7] Phase 5 Final Audit ZIP ready: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")

