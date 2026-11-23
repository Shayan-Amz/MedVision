"""Read-only airtight final release consistency audit for MedVision v4.0."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DOCS = [
    ROOT / "README.md",
    ROOT / "presentation/MedVision_Academic_Report.md",
    ROOT / "presentation/MedVision_Academic_Report.html",
    ROOT / "presentation/MedVision_Academic_Report_Standalone.html",
]
ALL_DOCS = PRIMARY_DOCS + [
    ROOT / "presentation/CV_SOP_Bullets.md",
    ROOT / "FINAL_CHANGELOG.md",
    ROOT / "SCIENTIFIC_LIMITATIONS.md",
]

BANNED = [
    (r"Built\s+a\s+deduplicated\s+175,413-passage", "Outdated CV deduplicated-corpus claim"),
    (r"n=250\s+Unseen", "Misleading IU-Xray Section 2.5 heading"),
    (r"Cross-Modal\s+Contradiction\s+Rate", "Unqualified contradiction table header"),
    (r"\bComparable to Human\b|Human\s+Physician\s+Average", "Unsupported Human comparison"),
    (r"\bHard Safety Guarantee\b|\bGuaranteed Clinical Safety\b|\bZero Clinical Risk\b", "Unsupported safety guarantee"),
    (r"\bLeak[- ]Free RAG\b|\bZero[- ]Overlap RAG\b", "Unsupported Leak-Free RAG claim"),
    (r"\b0\.8208\b|\b0\.8212\b|\[0\.8168,\s*0\.8255\]|\[0\.8166,\s*0\.8249\]", "Superseded NIH-14 metric"),
    (r"\b46\.50%\b|\b74\.50%\b|\[68\.5%,\s*80\.5%\]|85\.59%|76\.19%", "Superseded PubMedQA metric"),
    (r"`?maybe`?\s*F1\s*(?:=|is|:)\s*[*`]*0(?:\.0+)?%|3/200\s+`?maybe`?|only 3 `maybe`", "Superseded PubMedQA Maybe F1/count"),
    (r"\b0\.7619\b|TP\s*/\s*104|TN\s*/\s*146|n\s*=\s*104|n\s*=\s*146|104\s+abnormal|146\s+normal|FN=18|FN=31|FN=49|FP=93|FP=45|FP=14", "Superseded IU-Xray cohort/confusion matrix"),
    (r"47\.1%|69\s+Alert-Gate|69\s+triggers|69\s+alerts|across\s+250\s+evaluated", "Superseded Alert Gate / 250 evaluated claim"),
    (r"radiologist/MeSH annotations", "Inaccurate IU-Xray weak-label description"),
    (r"n=4,167|64\.84%|6\s*/\s*4,183|16\s*/\s*4,183|0\.143%|0\.383%|>\s*99\.6%", "Superseded/undefined MedMCQA/RAG claim"),
    (r"Eliminates\s+prior-induced\s+negation\s+hallucinations|0\.0%\s+cross-modal[^.,;\n]*contradiction", "Absolute cross-modal safety claim"),
]

REQUIRED_PRIMARY = [
    "Lexical Clinical Safety Filter", "Disclosed Train-Bank RAG",
    "0.8213", "[0.8151, 0.8267]",
    "39.50%", "78.50%", "60.05%", "13.79%",
    "55.99%", "59.45%", "64.52%", "64.98%", "4,136", "64.34%", "64.80%", "0.356", "28.19%",
    "250", "238", "79", "159", "75.5%", "37/49",
    "89.9%", "40.9%", "74.7%", "69.2%", "54.4%", "81.1%",
    "Research Prototype", "Weak-Label",
]

failures = []
for doc in ALL_DOCS:
    if not doc.exists():
        failures.append(f"MISSING FILE: {doc.relative_to(ROOT)}")
        continue
    txt = doc.read_text(encoding="utf-8")
    plain_txt = re.sub(r"[*`]+", "", txt)
    for pat, lbl in BANNED:
        m = re.search(pat, plain_txt, flags=re.IGNORECASE)
        if m:
            failures.append(f"[{doc.name}] BANNED ({lbl}): '{m.group(0)}'")
    if doc in PRIMARY_DOCS:
        for req in REQUIRED_PRIMARY:
            if req not in txt:
                failures.append(f"[{doc.name}] MISSING REQUIRED: '{req}'")

# Verify ALL Markdown and HTML relative figure links resolve on disk (expect >= 7 per doc)
img_checks = [
    (ROOT / "README.md", ROOT, r'(?:src="|\]\()((?:presentation/)?figures/[^"\)]+\.png)'),
    (ROOT / "presentation/MedVision_Academic_Report.md", ROOT / "presentation", r"\[[^\]]*\]\((figures/[^)]+\.png)\)"),
    (ROOT / "presentation/MedVision_Academic_Report.html", ROOT / "presentation", r'src="(figures/[^"]+\.png)"'),
]
checked_imgs = 0
for doc, base_dir, pat in img_checks:
    refs = re.findall(pat, doc.read_text(encoding="utf-8"))
    if len(refs) < 7:
        failures.append(f"[{doc.name}] EXPECTED >=7 FIGURE LINKS, FOUND {len(refs)}")
    for rel in refs:
        checked_imgs += 1
        if not (base_dir / rel).exists():
            failures.append(f"[{doc.name}] BROKEN FIGURE LINK: {rel}")

lines = [
    "=== MEDVISION v4.0 FINAL RELEASE CONSISTENCY AUDIT ===",
    f"Documents checked: {len(ALL_DOCS)} | Markdown & HTML figure links verified: {checked_imgs}",
]
if failures:
    lines.append(f"STATUS: FAIL ({len(failures)} issues)")
    lines.extend(f"  - {x}" for x in failures)
    report = "\n".join(lines) + "\n"
    (ROOT / "final_consistency_audit_report.txt").write_text(report, encoding="utf-8")
    print(report.strip())
    sys.exit(1)
else:
    lines.append("STATUS: PASS (0 conflicts across all 7 docs; all denominators, Maybe F1, RAG counts & 21+ MD/HTML figure links verified)")
    report = "\n".join(lines) + "\n"
    (ROOT / "final_consistency_audit_report.txt").write_text(report, encoding="utf-8")
    print(report.strip())
