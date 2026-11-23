import json
from app.services.clinical_triage_engine import compute_multimodal_triage

scenarios = [
    ("1. Checkup Tumor (Mass 74%) — NO SpO2, NO Age", [{"name":"Mass","prob":0.74,"positive":True}], 0.44, "", ""),
    ("2. Pneumothorax (81%) — NO SpO2, NO Age", [{"name":"Pneumothorax","prob":0.81,"positive":True}], 0.46, None, None),
    ("3. Moderate Finding (Effusion 58%) + Severe Hypoxia (SpO2=88%)", [{"name":"Effusion","prob":0.58,"positive":True}], 0.52, "88", "70"),
    ("4. Healthy Checkup (Max 38%) — NO SpO2, NO Age", [{"name":"Infiltration","prob":0.38,"positive":False}], 0.41, "", ""),
]
for title, fnds, ctr, spo2, age in scenarios:
    res = compute_multimodal_triage(fnds, ctr, spo2_raw=spo2, age_raw=age)
    print(f"\n[{title}]")
    print(f"  Tier: {res['triage_tier']} | Urgency: {res['urgency']} | RiskScore: {res['clinical_risk_score']}/100 | Email n8n: {res['dispatch_email']}")
    print(f"  Rationale: {res['risk_rationale'][0]}")
