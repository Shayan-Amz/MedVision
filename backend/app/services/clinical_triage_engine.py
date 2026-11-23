from typing import Dict, Any, List, Optional

# Clinical severity tiers (ACR Critical Findings inspired)
STAT_RED_FLAGS = {
    "Pneumothorax": 0.72,
    "Edema": 0.72,
    "Effusion": 0.78,
    "Consolidation": 0.78,
}
ONCOLOGY_RED_FLAGS = {
    "Mass": 0.68,
}
ACUTE_PATHOLOGIES = {"Pneumothorax", "Edema", "Effusion", "Consolidation", "Pneumonia", "Atelectasis", "Infiltration"}
CHRONIC_LOW_URGENCY = {"Fibrosis", "Pleural_Thickening", "Hernia", "Emphysema"}


def _parse_optional_float(val: Any, min_v: float, max_v: float) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().replace("%", "")
    if not s or s.lower() in ("n/a", "na", "none", "unknown", "-"):
        return None
    try:
        num = float(s)
        if min_v <= num <= max_v:
            return num
    except ValueError:
        pass
    return None


def compute_multimodal_triage(
    findings: List[Dict[str, Any]],
    ctr_val: Optional[float],
    spo2_raw: Any = None,
    age_raw: Any = None,
) -> Dict[str, Any]:
    """
    Two-Layer Multimodal Clinical Risk & n8n Escalation Engine:
      Layer 1: Hard Red-Flag Override (Missing vitals NEVER suppress Pneumothorax, Edema, Effusion, or Mass).
      Layer 2: Missing-Aware Adaptive Risk Score (0-100) fusing Vision + CTR + SpO2 + Age + Multi-Pathology Synergy.
    """
    prob_map = {f["name"]: float(f["prob"]) for f in findings}
    pos_acute = [f["name"] for f in findings if f.get("positive") and f["name"] in ACUTE_PATHOLOGIES]
    top_f = findings[0] if findings else {"name": "None", "prob": 0.0}

    spo2 = _parse_optional_float(spo2_raw, 40.0, 100.0)
    age = _parse_optional_float(age_raw, 0.0, 120.0)

    rationale: List[str] = []

    # --- LAYER 1: HARD RED-FLAG OVERRIDE (Never suppressed by missing vitals) ---
    stat_triggers = [f"{k} ({prob_map[k]*100:.1f}%)" for k, thr in STAT_RED_FLAGS.items() if prob_map.get(k, 0.0) >= thr]
    onco_triggers = [f"{k} ({prob_map[k]*100:.1f}%)" for k, thr in ONCOLOGY_RED_FLAGS.items() if prob_map.get(k, 0.0) >= thr]

    # --- LAYER 2: MISSING-AWARE ADAPTIVE RISK SCORING (0-100) ---
    earned = 0.0
    available_max = 0.0

    # Signal 1: Vision Severity (max 45 pts - always available)
    available_max += 45.0
    max_acute_p = max([prob_map.get(k, 0.0) for k in ACUTE_PATHOLOGIES], default=0.0)
    max_mass_p = prob_map.get("Mass", 0.0)
    max_chronic_p = max([prob_map.get(k, 0.0) for k in CHRONIC_LOW_URGENCY], default=0.0)
    vision_signal = max(max_acute_p, max_mass_p, max_chronic_p * 0.55)
    if vision_signal >= 0.75:
        v_pts = 45.0
    elif vision_signal >= 0.60:
        v_pts = 32.0
    elif vision_signal >= 0.48:
        v_pts = 16.0
    else:
        v_pts = 0.0
    earned += v_pts
    rationale.append(f"Vision Primary: {top_f['name']} ({top_f['prob']*100:.1f}%) [+{v_pts:.0f}/45]")

    # Signal 2: Anatomical CTR Geometry (max 15 pts - when PSPNet CTR available)
    if ctr_val is not None:
        available_max += 15.0
        chf_synergy = (prob_map.get("Edema", 0.0) >= 0.50 or prob_map.get("Effusion", 0.0) >= 0.55)
        if ctr_val >= 0.55 and chf_synergy:
            c_pts = 15.0
            rationale.append(f"Anatomical CTR={ctr_val:.3f} + CHF/Fluid Synergy [+15/15]")
        elif ctr_val > 0.50:
            c_pts = 8.0
            rationale.append(f"Anatomical Cardiomegaly CTR={ctr_val:.3f} [+8/15]")
        else:
            c_pts = 0.0
        earned += c_pts

    # Signal 3: Multi-Pathology Synergy (max 10 pts - always available)
    available_max += 10.0
    if len(pos_acute) >= 2:
        earned += 10.0
        rationale.append(f"Multi-Pathology Acute Synergy ({len(pos_acute)} findings) [+10/10]")
    elif len(pos_acute) == 1:
        earned += 4.0

    # Signal 4: Physiological SpO2 (max 25 pts - ONLY included in denominator if provided!)
    hypoxia_escalator = False
    if spo2 is not None:
        available_max += 25.0
        if spo2 < 90.0:
            earned += 25.0
            hypoxia_escalator = True
            rationale.append(f"Severe Hypoxia SpO2={spo2:.0f}% (NEWS2 Red) [+25/25]")
        elif spo2 <= 93.0:
            earned += 15.0
            rationale.append(f"Moderate Hypoxia SpO2={spo2:.0f}% [+15/25]")
        else:
            rationale.append(f"SpO2={spo2:.0f}% (Normal) [+0/25]")
    else:
        rationale.append("SpO2: Not Provided (Adaptive Normalization — Imaging + CTR Only)")

    # Signal 5: Patient Age (max 5 pts - ONLY included in denominator if provided!)
    if age is not None:
        available_max += 5.0
        if age >= 65.0:
            earned += 5.0
            rationale.append(f"Age={age:.0f}y (>=65 risk factor) [+5/5]")
    else:
        rationale.append("Age: Not Provided")

    risk_score = round((earned / max(available_max, 1.0)) * 100.0, 1)

    # --- FINAL ESCALATION ROUTING DECISION ---
    override_reason = None
    if stat_triggers:
        triage_tier = "STAT_ESCALATION"
        urgency_label = "STAT"
        dispatch_email = True
        override_reason = f"Hard STAT Red-Flag Override: {', '.join(stat_triggers)}"
        risk_score = max(risk_score, 85.0)
    elif hypoxia_escalator and vision_signal >= 0.48:
        triage_tier = "STAT_ESCALATION"
        urgency_label = "STAT"
        dispatch_email = True
        override_reason = f"Physiological Escalation: Severe Hypoxia (SpO2={spo2:.0f}%) + Positive Imaging"
        risk_score = max(risk_score, 80.0)
    elif onco_triggers:
        triage_tier = "URGENT_ONCOLOGY_ALERT"
        urgency_label = "STAT"
        dispatch_email = True
        override_reason = f"Hard Oncology Red-Flag Override: {', '.join(onco_triggers)} (Never suppressed by missing/normal vitals)"
        risk_score = max(risk_score, 78.0)
    elif risk_score >= 65.0:
        triage_tier = "STAT_ESCALATION"
        urgency_label = "STAT"
        dispatch_email = True
    elif risk_score >= 40.0:
        triage_tier = "PRIORITY_QUEUE"
        urgency_label = "HIGH"
        dispatch_email = False
    else:
        triage_tier = "ROUTINE_ARCHIVE"
        urgency_label = "ROUTINE"
        dispatch_email = False

    if override_reason:
        rationale.insert(0, override_reason)

    return {
        "clinical_risk_score": risk_score,
        "triage_tier": triage_tier,
        "urgency": urgency_label,
        "dispatch_email": dispatch_email,
        "red_flag_override": override_reason,
        "vitals_provided": {"spo2": spo2 is not None, "age": age is not None},
        "risk_rationale": rationale,
    }
