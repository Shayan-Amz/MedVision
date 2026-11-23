import base64, json, os, re, zipfile
from pathlib import Path
from scipy.stats import binomtest

# 1. Fix underflowed mcnemar_exact_p=0.0 in pubmedqa_audit.json
pqa_path = Path("./checkpoints/pubmedqa_audit.json")
pqa = json.load(open(pqa_path, encoding="utf-8"))
mc_p = float(binomtest(min(pqa["mcnemar_b01"], pqa["mcnemar_b10"]), pqa["mcnemar_b01"] + pqa["mcnemar_b10"], 0.5).pvalue)
pqa["mcnemar_exact_p"] = float(f"{mc_p:.6e}")
pqa["mcnemar_p_display"] = f"{mc_p:.2e} (< 0.0001)"
pqa_path.write_text(json.dumps(pqa, indent=2), encoding="utf-8")

# 2. Sync medvision_nih14_ensemble_meta.json, main.py, index.html, test_medvision_api.py to 0.8213
meta_path = Path("./checkpoints/medvision_nih14_ensemble_meta.json")
meta = json.load(open(meta_path, encoding="utf-8"))
meta["test_mean_auroc_densenet"] = 0.8139
meta["test_mean_auroc_convnext"] = 0.8141
meta["test_mean_auroc_ensemble"] = 0.8213
meta["n_official_test"] = 25596
meta["n_unique_test_patients"] = 2797
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

for fstr in ("./app/main.py", "./app/static/index.html", "./tests/test_medvision_api.py"):
    fp = Path(fstr)
    if fp.exists():
        txt = fp.read_text(encoding="utf-8")
        txt = txt.replace("0.8208", "0.8213").replace("0.8212", "0.8213")
        fp.write_text(txt, encoding="utf-8")

# 3. Sync MedVision_Academic_Report.html with all verified Phase 2/3 metrics
nih_pc = json.load(open("./checkpoints/nih14_full_25596_per_class.json", encoding="utf-8"))
html_path = Path("../presentation/MedVision_Academic_Report.html")
h = html_path.read_text(encoding="utf-8")

h = re.sub(
    r'<div class="kpi"><div class="lbl">NIH-14 Official Test</div>.*?</div></div>',
    '<div class="kpi"><div class="lbl">NIH-14 Official Test</div><div class="val">0.8213 AUROC</div><div class="sub">n=25,596 (2,797 pts) | 14/14 best | &Delta;+0.0074 vs DN</div></div>',
    h, count=1
)
h = re.sub(
    r'<div class="kpi"><div class="lbl">PubMedQA Unseen.*?</div>.*?</div></div>',
    '<div class="kpi"><div class="lbl">PubMedQA Unseen (n=200)</div><div class="val">78.50%</div><div class="sub">Fair Logit Base 39.50% (+39.0 pp, p=1.29e-14) | Macro-F1 60.05%</div></div>',
    h, count=1
)
h = re.sub(
    r'<div class="kpi"><div class="lbl">MedMCQA.*?</div>.*?</div></div>',
    '<div class="kpi"><div class="lbl">MedMCQA Validation</div><div class="val">59.45% / 64.80%</div><div class="sub">Closed-Book (n=4,183) | Deduped Train-Bank RAG (n=4,136)</div></div>',
    h, count=1
)

sorted_cls = sorted(nih_pc["per_class"], key=lambda r: r["ens"], reverse=True)
rows = [
    f"      <tr><td><b>{r['class']}</b></td><td>{r['dn']:.4f}</td><td>{r['cn']:.4f}</td>"
    f"<td style='color:#059669;font-weight:700'>{r['ens']:.4f}</td>"
    f"<td>{r['sens']:.1f}%</td><td>{r['spec']:.1f}%</td><td>{r['fpr']:.1f}%</td><td>{r['fnr']:.1f}%</td></tr>"
    for r in sorted_cls
]
rows.append(
    f'      <tr style="background:#e2e8f0;font-weight:800"><td>MEAN (14 Classes, n=25,596)</td>'
    f'<td>{nih_pc["mean_dn"]:.4f}</td><td>{nih_pc["mean_cn"]:.4f}</td><td style="color:#059669">{nih_pc["mean_ens"]:.4f}</td>'
    f'<td>{nih_pc["mean_sens"]:.1f}%</td><td>{nih_pc["mean_spec"]:.1f}%</td>'
    f'<td>{100.0 - nih_pc["mean_spec"]:.1f}%</td><td>{100.0 - nih_pc["mean_sens"]:.1f}%</td></tr>'
)
h = re.sub(r"<tbody>(?:(?!</tbody)[\s\S])*?MEAN \(14 Classes[\s\S]*?</tbody>", "<tbody>\n" + "\n".join(rows) + "\n    </tbody>", h, count=1)

h = re.sub(
    r'<div class="fig-cap">Figure 7:.*?</div>',
    '<div class="fig-cap">Figure 7: (Left) NIH-14 Official Test ablation (n=25,596; 2,797 unique patients) with Patient-Level Cluster Paired Bootstrap (B=2,000, seed=42): Ensemble 0.8213 [0.8151, 0.8267], Paired &Delta;AUROC +0.0074 [+0.0055, +0.0096] vs DenseNet &amp; +0.0072 [+0.0055, +0.0090] vs ConvNeXt. (Right) Weak-Label External Feasibility Evaluation on IU-Xray (n=238 unseen frontal CXRs from 250 UID-disjoint studies; AUROC 0.7408 [0.6699, 0.8070], 0/238 lexical contradictions [0, 1.54%], Safety Guardrail: 13&rarr;0 malignancy &amp; 8&rarr;0 biopsy directives, 2.53s, 6.52 GB VRAM).</div>',
    h, count=1
)

sec6_tbody = """    <tbody>
      <tr><td><b>NIH-14 Official Test (n=25,596; 2,797 pts)</b></td><td>Wang et al. 0.7450 | DenseNet TTA 0.8139 | ConvNeXt TTA 0.8141</td><td style="color:#059669;font-weight:700">0.8213 Ensemble (14/14 best)</td><td>Patient-Cluster CI [0.8151, 0.8267] | Paired &Delta;AUROC +0.0074 [+0.0055, +0.0096] vs DN; +0.0072 [+0.0055, +0.0090] vs CN</td></tr>
      <tr><td><b>PubMedQA Unseen (n=200; 0 RAG; 0 Overlap)</b></td><td>Fair Logit Base 39.50% [32.98, 46.41] (Macro-F1 33.51%)</td><td style="color:#059669;font-weight:700">78.50% Fair Logit (Macro-F1 60.05%)</td><td>Wilson CI [72.30, 83.63] | Exact McNemar p = 1.29&times;10<sup>&minus;14</sup> (b01=94, b10=16) | maybe F1 13.79%</td></tr>
      <tr><td><b>MedMCQA Validation (n=4,183; Deduped n=4,136)</b></td><td>Base Closed-Book 55.99% | v3 Closed-Book (0 RAG) 59.45%</td><td style="color:#059669;font-weight:700">64.52% Single / 64.98% Cyclic-4 (Deduped n=4,136: 64.34% / 64.80%)</td><td>Train-Bank Explanation/Flashcard RAG disclosed (gold opt in ctx 28.19% vs any distractor 32.70% / 15.92% per slot; explicit Q&amp;A key 51/4,183 = 1.22%)</td></tr>
      <tr><td><b>Weak-Label External IU-Xray (n=238; 0 UID Overlap)</b></td><td>Report-Heuristic Labels (96/238 disagreement vs Problems, &kappa;=0.275)</td><td style="color:#059669;font-weight:700">AUROC 0.7408 [0.6699, 0.8070] | Triage 89.9/40.9% | Bal. 74.7/69.2% | Alert 54.4/81.1%</td><td>0/238 lexical contradictions &amp; 0/238 OOD rejects (95% CI [0, 1.54%]) | Safety Guardrail: 13&rarr;0 malignancy, 8&rarr;0 biopsy (13 sanitized) | 2.53s, 6.52 GB</td></tr>
    </tbody>"""
h = re.sub(r'    <tbody>\s*<tr><td><b>NIH-14 Official Test[\s\S]*?</tbody>', sec6_tbody, h, count=1)

new_callout = """  <div class="callout" style="margin-top:14px;">
    <b>Verified Evaluation Protocols, Provenance &amp; Clinical Safety:</b><br/>
    &bull; <b>NIH-14 Patient-Level Cluster Bootstrap:</b> Resampling over <b>2,797 unique patients</b> (25,596 images, B=2,000, seed=42) yields Paired <code>&Delta;AUROC = +0.0074 [0.0055, 0.0096]</code> vs DenseNet-121 and <code>+0.0072 [0.0055, 0.0090]</code> vs ConvNeXt-Small (both strictly &gt; 0).<br/>
    &bull; <b>PubMedQA Zero-Overlap Protocol:</b> Evaluated on <code>shuffle(seed=42)[0:200]</code> strictly disjoint from <code>phase2a</code> train <code>[200:1000]</code> (0 <code>pubid</code> overlap, 0 RAG), with Fair Logit scoring (<code>39.50% &rarr; 78.50%</code>, +39.0 pp, Macro-F1 <code>60.05%</code>, Exact McNemar <code>p = 1.29&times;10<sup>&minus;14</sup></code>).<br/>
    &bull; <b>MedMCQA Validation &amp; RAG Provenance Disclosure:</b> Clean parametric closed-book gain is <code>55.99% &rarr; 59.45%</code> (0 RAG). The 175,413-passage FAISS index is built from <b>MedMCQA training-split explanations (with textbook citations) and clinical flashcards</b>: gold option appears verbatim in <code>28.19%</code> (<code>1,179/4,183</code>; Cyclic-4 acc <code>84.14%</code> vs <code>57.46%</code> when absent), any distractor in <code>32.70%</code> (<code>15.92%</code> per distractor slot), and explicit Q&amp;A answer-key strings in <code>51/4,183 (1.22%)</code>. After removing 47 exact/8-gram overlaps (<code>n=4,136</code>), Train-Bank RAG achieves <code>64.34%</code> Single / <code>64.80%</code> Cyclic-4 and is reported transparently as <b>Train-Bank Retrieval-Augmented</b> performance rather than a semantically leak-free textbook-only benchmark.<br/>
    &bull; <b>Post-Generation Clinical Safety Guardrail (n=238 Audit):</b> Automated safety auditing of all 238 IU-Xray LLM drafts identified <code>13/238</code> unverified malignancy claims and <code>8/238</code> direct biopsy directives (<code>13</code> reports sanitized); installing <code>apply_clinical_safety_guardrail()</code> in production reduced both to <b><code>0/238</code></b>.
  </div>"""
h = re.sub(r'  <div class="callout" style="margin-top:14px;">[\s\S]*?</div>', new_callout, h, count=1)

new_sec7 = """  <h2>7. Transparent Scientific Limitations, Statistical Precision &amp; Clinical Scope</h2>
  <ul>
    <li><b>Weak-Label External Feasibility Scope (IU-Xray, n=238):</b> From the 250 UID-disjoint held-out studies (<code>shuffle(seed=42)[:250]</code>, 0 overlap with <code>radiologist</code> LoRA train), 12 lacked frontal radiographs, yielding <b><code>n=238</code></b> evaluated cases. Binary labels derived from report-impression heuristics disagreed with Open-I <code>Problems</code> annotations on <b><code>96/238 (40.34%, Cohen&rsquo;s &kappa; = 0.275)</code></b> (acute vs. chronic findings). Accordingly, this is strictly a <b>Weak-Label External Feasibility Evaluation</b>; multi-reader blinded adjudication and multi-center validation (CheXpert/PadChest) are required before clinical claims.</li>
    <li><b>Zero-Event Statistical Precision (0/238):</b> Observing <code>0/238</code> lexical negation contradictions or frontal OOD false rejections yields an exact Clopper&ndash;Pearson 95% CI of <b><code>[0.0%, 1.54%]</code></b>, not a zero population error rate; lexical checks do not substitute for full clinical NLI.</li>
    <li><b>Operating-Point Trade-offs &amp; Non-Diagnostic Scope:</b> External operating points exhibit clear sensitivity&ndash;specificity trade-offs (ER Triage: <code>89.9% / 40.9%</code>; Balanced: <code>74.7% / 69.2%</code>; Alert Gate: <code>54.4% / 81.1%, PPV 58.9%</code>). MedVision AI Studio v4.0 is a <b>Research Prototype &amp; CDSS Proof-of-Concept</b>&mdash;not a cleared diagnostic or clinical triage device&mdash;and requires prospective multi-center trials and regulatory review prior to clinical use.</li>
    <li><b>RAG Provenance &amp; PubMedQA Class Imbalance:</b> MedMCQA RAG uses a training-split explanation/flashcard corpus (disclosed above); PubMedQA <code>maybe</code> F1 remains low (<code>13.79%</code>) despite <code>78.50%</code> accuracy and <code>60.05%</code> Macro-F1.</li>
    <li><b>Small-Lesion Localization &amp; Mask Definition:</b> 256&times;256 classification yields coarse Grad-CAM++ localization for millimeter nodules (<code>2.5%</code> Pointing Game); Montgomery manual masks exclude retrocardiac left-lung pixels (Left Dice <code>83.26%</code> vs Right <code>90.36%</code>).</li>
    <li><b>Continual DPO PoC:</b> The 32-pair DPO cycle validates the human-in-the-loop retraining pipeline rather than claiming offline DPO benchmark gains.</li>
  </ul>"""
h = re.sub(r'  <h2>7\. Transparent Scientific Limitations[\s\S]*?</ul>', new_sec7, h, count=1)

html_path.write_text(h, encoding="utf-8")

pres_dir = html_path.parent
def embed_img(m):
    fp = pres_dir / m.group(1)
    return f'src="data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode("ascii")}"' if fp.exists() else m.group(0)
standalone = re.sub(r'src="(figures/[^"]+\.png)"', embed_img, h)
(pres_dir / "MedVision_Academic_Report_Standalone.html").write_text(standalone, encoding="utf-8")

for mstr in ("../presentation/MedVision_Academic_Report.md", "../README.md"):
    mp = Path(mstr)
    if mp.exists():
        t = mp.read_text(encoding="utf-8")
        t = t.replace("0.8212", "0.8213").replace("[0.8168, 0.8255]", "[0.8151, 0.8267]")
        mp.write_text(t, encoding="utf-8")

zip_path = Path(os.environ["USERPROFILE"]) / "Desktop" / "MedVision_Audit_Phase3_Final.zip"
bundle = [
    ("backend/audit_step1_nih14_25596_exact.py", "./audit_step1_nih14_25596_exact.py"),
    ("backend/audit_step2_pubmedqa_zero_leak.py", "./audit_step2_pubmedqa_zero_leak.py"),
    ("backend/audit_step3_iuxray_zero_leak.py", "./audit_step3_iuxray_zero_leak.py"),
    ("backend/audit_step4_medmcqa_csv.py", "./audit_step4_medmcqa_csv.py"),
    ("backend/audit_step5_rag_provenance_and_manifest.py", "./audit_step5_rag_provenance_and_manifest.py"),
    ("backend/audit_step6_safety_guardrail.py", "./audit_step6_safety_guardrail.py"),
    ("backend/phase2a_scaled_multitask_sft.py", "./phase2a_scaled_multitask_sft.py"),
    ("backend/phase6d_sync_25596_exact_ci_and_figs.py", "./phase6d_sync_25596_exact_ci_and_figs.py"),
    ("backend/phase7_patient_cluster_bootstrap_and_html_sync.py", "./phase7_patient_cluster_bootstrap_and_html_sync.py"),
    ("backend/app/services/clinical_llm_rag.py", "./app/services/clinical_llm_rag.py"),
    ("backend/checkpoints/medvision_nih14_ensemble_meta.json", "./checkpoints/medvision_nih14_ensemble_meta.json"),
    ("backend/checkpoints/pubmedqa_audit.json", "./checkpoints/pubmedqa_audit.json"),
    ("backend/checkpoints/pubmedqa_split_manifest.json", "./checkpoints/pubmedqa_split_manifest.json"),
    ("backend/checkpoints/medmcqa_rag_provenance_and_options_audit.json", "./checkpoints/medmcqa_rag_provenance_and_options_audit.json"),
    ("backend/checkpoints/clinical_safety_guardrail_audit.json", "./checkpoints/clinical_safety_guardrail_audit.json"),
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

print(f"ALL SYNCED & Phase 3 Final ZIP ready: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")
