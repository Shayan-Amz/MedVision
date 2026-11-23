import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "0.8213 AUROC" in body["vision_backbone"]

def test_ui_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "0.8213 Mean AUROC" in r.text
    assert "64.98%" in r.text

def test_ood_guardrail_rejects_non_xray():
    fake = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    fake.save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/api/v1/studies/upload-analyze",
        files={"file": ("selfie.png", buf, "image/png")},
        data={"patient_name": "Test OOD", "patient_mrn": "MRN-OOD"},
    )
    assert r.status_code == 400
    assert "OOD Guardrail" in r.json()["detail"]

def test_studies_history_and_dpo_endpoints():
    r1 = client.get("/api/v1/studies/history")
    assert r1.status_code == 200
    r2 = client.get("/api/v1/dpo/list")
    assert r2.status_code == 200
    r3 = client.get("/api/v1/training/status")
    assert r3.status_code == 200

if __name__ == "__main__":
    test_health_endpoint()
    test_ui_serves_html()
    test_ood_guardrail_rejects_non_xray()
    test_studies_history_and_dpo_endpoints()
    print("[ALL 4 AUTOMATED API & OOD TESTS PASSED!]")
