import json, py_compile, re
from pathlib import Path

rag_path = Path("./app/services/clinical_llm_rag.py")
src = rag_path.read_text(encoding="utf-8")

guardrail_fn = '''
def apply_clinical_safety_guardrail(report: str):
    """Post-generation clinical safety guardrail: blocks unverified malignancy diagnoses and direct invasive directives."""
    flags = []
    safe = report or ""
    mal_pat = re.compile(r"\\b(malignan(?:cy|t)|carcinoma|cancer(?:ous)?|metastat(?:ic|sis)|neoplasm)\\b", re.IGNORECASE)
    if mal_pat.search(safe):
        safe = mal_pat.sub("indeterminate lesion (radiologist correlation required)", safe)
        flags.append("malignancy_claim_neutralized")
    inv_pat = re.compile(r"\\b((?:immediate|urgent|stat)\\s+)?(biopsy|surgical\\s+resection|surgery|hospitalization|admit(?:ting)?\\s+immediately)\\b", re.IGNORECASE)
    if inv_pat.search(safe):
        safe = inv_pat.sub("attending physician evaluation and follow-up imaging as clinically indicated", safe)
        flags.append("invasive_directive_neutralized")
    if flags and "Decision-support draft only" not in safe:
        safe = safe.rstrip() + " [Safety Guardrail: Decision-support draft only; requires attending radiologist verification prior to diagnostic labeling or invasive workup.]"
    return safe, flags
'''

if "def apply_clinical_safety_guardrail" not in src:
    target = '    return {"report": text, "adapter_used": adapter}'
    assert target in src, "Target return line not found in clinical_llm_rag.py"
    replacement = (
        '    text, safety_flags = apply_clinical_safety_guardrail(text)\n'
        '    return {"report": text, "adapter_used": adapter, "safety_guardrail_flags": safety_flags}'
    )
    src = src.replace(target, replacement, 1)
    src = guardrail_fn + "\n" + src
    if "import re" not in src.splitlines()[:25]:
        src = "import re\n" + src
    rag_path.write_text(src, encoding="utf-8")

py_compile.compile(str(rag_path), doraise=True)

from app.services.clinical_llm_rag import apply_clinical_safety_guardrail
iu = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
mal_re = re.compile(r"\b(malignan(?:cy|t)|carcinoma|cancer(?:ous)?|metastat(?:ic|sis)|neoplasm)\b", re.IGNORECASE)
bio_re = re.compile(r"\b(biopsy|surgical\s+resection|surgery|hospitalization)\b", re.IGNORECASE)

before_mal = before_bio = after_mal = after_bio = total_flagged = 0
examples = []
for it in iu["items"]:
    raw = it.get("llm_report", "")
    bm, bb = bool(mal_re.search(raw)), bool(bio_re.search(raw))
    safe, flags = apply_clinical_safety_guardrail(raw)
    am, ab = bool(mal_re.search(safe)), bool(bio_re.search(safe))
    before_mal += int(bm); before_bio += int(bb)
    after_mal += int(am); after_bio += int(ab)
    if flags:
        total_flagged += 1
        if str(it["uid"]) == "744" or len(examples) < 5:
            examples.append({"uid": it["uid"], "flags": flags, "before": raw, "after": safe})

audit_out = {
    "n_evaluated": len(iu["items"]),
    "before_guardrail": {"malignancy_claims": before_mal, "invasive_biopsy_directives": before_bio},
    "after_guardrail": {"malignancy_claims": after_mal, "invasive_biopsy_directives": after_bio},
    "total_reports_sanitized": total_flagged,
    "examples": examples,
}
with open("./checkpoints/clinical_safety_guardrail_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit_out, f, indent=2)
print("SAFETY GUARDRAIL INSTALLED & AUDITED:", json.dumps({k: v for k, v in audit_out.items() if k != "examples"}, indent=2))
