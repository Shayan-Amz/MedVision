import json
import re
from pathlib import Path
import base64, io, json, os, time, threading
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from PIL import Image
import pydicom
import numpy as np
import torch

from app.models.schema import PatientCreate, PhysicianFeedbackRequest, TrainingJobConfig
from app.services import local_db
from app.services.n8n_orchestrator import N8nOrchestrator
n8n_service = N8nOrchestrator()
from app.services.vision_ensemble import analyze_chest_xray_ensemble
from app.services.clinical_triage_engine import compute_multimodal_triage
from app.services.clinical_llm_rag import (
    synthesize_radiology_impression,
    answer_clinical_followup_with_rag,
)

app = FastAPI(
    title="MedVision & Clinical LLM Studio v4.0",
    description="Multimodal Thoracic Radiology Ensemble (0.8213 AUROC) + Dual-Adapter 7B Clinical LLM + 175k Reranked Medical RAG + Continual DPO",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_db.init_db()
DPO_FEEDBACK_VAULT: List[dict] = []
STUDY_CACHE: dict = {}
STUDY_CACHE_DIR = Path("./study_cache")
STUDY_CACHE_DIR.mkdir(exist_ok=True)
N8N_AUDIT_LOGS: List[dict] = []
TRAINING_STATE = {
    "status": "idle",
    "job_id": None,
    "pairs_exported": 0,
    "dataset_path": None,
    "last_updated": None,
    "message": "No continual DPO job running.",
}


class FollowupReq(BaseModel):
    question: str
    study_context: str = ""
    primary_finding: str = ""
    impression: str = ""


class ManualAlertReq(BaseModel):
    patient_name: str = "Unknown"
    patient_mrn: str = "MRN-000"
    patient_info: str = ""
    study_id: str = ""
    primary_finding: str = "Critical Finding"
    confidence_score: float = 0.85
    urgency: str = "STAT"
    impression: str = ""
    clinical_impression: str = ""


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_uploaded_image(raw_bytes: bytes, filename: str) -> Image.Image:
    if filename.lower().endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = (arr / arr.max()) * 255.0
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255.0 - arr
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _run_dpo_export_worker(job_id: str, epochs: int):
    TRAINING_STATE.update({
        "status": "running",
        "job_id": job_id,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "message": "Exporting physician-verified DPO pairs from SQLite...",
    })
    try:
        jsonl_text = local_db.export_all_dpo_jsonl()
        lines = [ln for ln in jsonl_text.splitlines() if ln.strip()]
        os.makedirs("./checkpoints/continual_dpo", exist_ok=True)
        out_path = f"./checkpoints/continual_dpo/dpo_batch_{job_id[:8]}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(jsonl_text)
        TRAINING_STATE.update({
            "status": "completed",
            "pairs_exported": len(lines),
            "dataset_path": out_path,
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "message": f"Exported {len(lines)} DPO pairs to {out_path} (ready for offline QLoRA DPO, epochs={epochs}).",
        })
    except Exception as e:
        TRAINING_STATE.update({
            "status": "failed",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "message": str(e),
        })


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_studio_ui():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vision_backbone": "NIH-14 Ensemble (0.55 DenseNet-121 + 0.45 ConvNeXt-Small, 2-View TTA, 0.8213 AUROC) + PSPNet (87.0% Lung Dice) + OOD Guardrail",
        "llm_backbone": "MedVision-7B Dual-Adapter (IU-Xray Radiologist 79.32% F1 + Clinical QA v2 78.5% PubMedQA) + CPU 175k Reranked RAG (64.98% MedMCQA)",
    }


@app.post("/api/v1/studies/upload-analyze")
async def upload_and_analyze_xray(
    file: UploadFile = File(...),
    patient_name: str = Form("Anonymous Patient"),
    patient_mrn: str = Form("MRN-0001"),
    age: str = Form("55"),
    spo2: str = Form("96"),
    clinical_context: str = Form("Routine chest radiograph"),
    operating_mode: str = Form("triage"),
):
    raw_bytes = await file.read()
    try:
        orig_pil = decode_uploaded_image(raw_bytes, file.filename or "scan.png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or DICOM file: {e}")

    t0 = time.perf_counter()
    vres = analyze_chest_xray_ensemble(orig_pil, operating_mode=operating_mode)
    vision_ms = round((time.perf_counter() - t0) * 1000, 1)

    if vres.get("ood_rejected"):
        raise HTTPException(
            status_code=400,
            detail=f"OOD Guardrail Rejected Image: {vres.get('ood_reason')} Please upload a valid chest radiograph (PNG/JPG/DICOM).",
        )

    primary = vres["primary_finding"]
    top_conf = vres["top_confidence"]
    ctr_val = vres["ctr"]
    top3 = vres["top3"]
    triage = compute_multimodal_triage(vres["findings"], ctr_val, spo2_raw=spo2, age_raw=age)
    urgency = triage["urgency"]

    patient_ctx = (
        f"Patient={patient_name} ({patient_mrn}), Age={age}, SpO2={spo2}%, "
        f"Context={clinical_context}, Exact PSPNet CTR={ctr_val if ctr_val is not None else 'N/A'}"
    )
    llm_res = synthesize_radiology_impression(
        primary=primary,
        top_conf=top_conf,
        top3=[(f['name'], f['prob']) for f in vres['findings'] if f['prob'] >= 0.45] or top3,
        urgency=urgency,
        patient_ctx=patient_ctx,
    )
    live_impression = llm_res["report"]
    study_id = str(uuid4())

    # Persist study into SQLite history
    try:
        local_db.save_study(
            patient_name=patient_name,
            patient_mrn=patient_mrn,
            age=str(age),
            spo2=str(spo2),
            context=clinical_context,
            primary_finding=primary,
            confidence=float(top_conf),
            urgency=urgency,
            ctr=str(ctr_val) if ctr_val is not None else "N/A",
            impression=live_impression,
        )
    except Exception:
        pass

    n8n_dispatch = None
    if triage["dispatch_email"]:
        n8n_dispatch = await n8n_service.trigger_emergency_triage(
            patient={"mrn": patient_mrn, "full_name": patient_name, "age": age, "vitals": {"spO2": f"{spo2}%"}},
            vision_result={
                "study_id": study_id,
                "primary_finding": primary,
                "confidence_score": top_conf,
                "urgency": "CRITICAL" if urgency == "STAT" else urgency,
        "clinical_risk_score": triage["clinical_risk_score"],
        "triage_tier": triage["triage_tier"],
        "risk_rationale": triage["risk_rationale"],
                "ctr_ratio": ctr_val,
                "roi_bounding_box": {"method": "Heart-Aware Lung-Gated Grad-CAM++"},
            },
            clinical_report={"generated_impression": live_impression, "recommended_actions": [f"Risk Score {triage['clinical_risk_score']}/100 ({triage['triage_tier']})", *triage["risk_rationale"][:3]]},
        )
        N8N_AUDIT_LOGS.append(n8n_dispatch)
        try:
            local_db.save_n8n_log(
                n8n_dispatch.get("workflow_name", "01_emergency_clinical_triage"),
                f"{primary} ({top_conf*100:.1f}%) | Risk {triage['clinical_risk_score']}/100 ({triage['triage_tier']})",
                n8n_dispatch.get("execution_status", "DISPATCHED"),
                datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            )
        except Exception:
            pass

    class_probs = {f["name"]: f["prob"] for f in vres["findings"][:6]}
    payload_out = {
        "study_id": study_id,
        "filename": file.filename,
        "device_used": f"{'CUDA' if torch.cuda.is_available() else 'CPU'} | Ensemble (0.8213 AUROC) + {llm_res['adapter_used']}",
        "latency_ms": vision_ms,
        "primary_finding": primary,
        "confidence_score": top_conf,
        "urgency": urgency,
        "operating_mode": operating_mode,
        "clinical_risk_score": triage["clinical_risk_score"],
        "triage_tier": triage["triage_tier"],
        "risk_rationale": triage["risk_rationale"],
        "ctr_ratio": ctr_val,
        "heart_preserved_in_cam": vres["heart_preserved_in_cam"],
        "lung_coverage_pct": vres["lung_coverage_pct"],
        "class_probabilities": class_probs,
        "all_findings": vres["findings"],
        "clinical_impression": live_impression,
        "adapter_used": llm_res["adapter_used"],
        "original_image_b64": pil_to_base64_png(orig_pil.resize((512, 512))),
        "heatmap_image_b64": pil_to_base64_png(vres["gradcam_image"]),
        "n8n_automation": n8n_dispatch,
    }
    STUDY_CACHE[study_id] = payload_out
    try:
        (STUDY_CACHE_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '', study_id)}.json").write_text(json.dumps(payload_out), encoding="utf-8")
    except Exception:
        pass
    return payload_out


@app.get("/api/v1/studies/history")
async def get_studies_history(q: str = "", page: int = 1, page_size: int = 10):
    return local_db.list_studies(q=q, page=page, page_size=page_size)


@app.get("/api/v1/studies/{study_id}")
async def get_study_by_id(study_id: str):
    if study_id in STUDY_CACHE:
        return STUDY_CACHE[study_id]
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", study_id)
    fp = STUDY_CACHE_DIR / f"{safe_id}.json"
    if fp.exists():
        data = json.loads(fp.read_text(encoding="utf-8"))
        STUDY_CACHE[study_id] = data
        return data
    raise HTTPException(status_code=404, detail="Study not found in cache or disk storage.")


@app.post("/api/v1/studies/analyze")
async def analyze_clinical_study(patient: PatientCreate):
    return {"status": "ok", "message": "Use /api/v1/studies/upload-analyze for real radiograph + live 7B LLM inference."}


@app.post("/api/v1/clinical/followup")
async def clinical_followup(req: FollowupReq):
    ctx = req.study_context or f"Primary: {req.primary_finding}. Impression: {req.impression}"
    res = answer_clinical_followup_with_rag(question=req.question, study_context=ctx)
    return {
        "answer": res["answer"],
        "adapter_used": res["adapter_used"],
        "evidence": res["evidence"],
    }


@app.post("/api/v1/feedback/dpo-pair")
async def submit_physician_dpo_feedback(feedback: PhysicianFeedbackRequest):
    DPO_FEEDBACK_VAULT.append(feedback.model_dump())
    local_db.save_dpo_pair(
        str(uuid4()),
        str(feedback.study_id),
        feedback.physician_name,
        "Clinical Impression",
        feedback.clinician_notes or "Raw 7B Draft",
        feedback.edited_report_text,
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    retrain_event = None
    if len(DPO_FEEDBACK_VAULT) >= 25:
        retrain_event = await n8n_service.trigger_auto_dpo_retraining(len(DPO_FEEDBACK_VAULT), 25)
        N8N_AUDIT_LOGS.append(retrain_event)
    return {"status": "saved", "vault_count": len(DPO_FEEDBACK_VAULT), "retrain_event": retrain_event}


@app.post("/api/v1/training/start")
async def start_qlora_dpo_job(config: TrainingJobConfig):
    job_id = str(uuid4())
    t = threading.Thread(target=_run_dpo_export_worker, args=(job_id, getattr(config, "epochs", 1)), daemon=True)
    t.start()
    return {"status": "started", "job_id": job_id, "message": "Continual DPO background worker triggered."}


@app.get("/api/v1/training/status")
async def get_training_status():
    return TRAINING_STATE


@app.websocket("/ws/training-telemetry")
async def training_telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json(TRAINING_STATE)
        await websocket.close()
    except WebSocketDisconnect:
        pass


@app.post("/api/v1/n8n/manual-alert")
async def manual_n8n_alert(req: ManualAlertReq):
    pname = req.patient_info or req.patient_name
    imp = req.clinical_impression or req.impression
    cached = STUDY_CACHE.get(req.study_id)
    if not cached and req.study_id:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", req.study_id)
        fp = STUDY_CACHE_DIR / f"{safe_id}.json"
        if fp.exists():
            cached = json.loads(fp.read_text(encoding="utf-8"))
    ctr_val = cached.get("ctr_ratio") if isinstance(cached, dict) and cached.get("ctr_ratio") is not None else "N/A"
    risk_val = cached.get("clinical_risk_score", 90.0) if isinstance(cached, dict) else 90.0
    actions = ["Clinician Manual STAT Alert"] + (list(cached.get("risk_rationale") or []) if isinstance(cached, dict) else [])
    dispatch = await n8n_service.trigger_emergency_triage(
        patient={"mrn": req.patient_mrn, "full_name": pname, "age": "N/A", "vitals": {"spO2": "N/A"}},
        vision_result={"study_id": req.study_id, "primary_finding": req.primary_finding, "confidence_score": req.confidence_score, "urgency": "CRITICAL", "triage_tier": "MANUAL_STAT_ALERT", "clinical_risk_score": risk_val, "ctr_ratio": ctr_val, "roi_bounding_box": {"method": "Manual Clinician STAT"}},
        clinical_report={"generated_impression": imp, "recommended_actions": actions},
    )
    N8N_AUDIT_LOGS.append(dispatch)
    try:
        local_db.save_n8n_log(
            dispatch.get("workflow_name", "01_emergency_clinical_triage"),
            f"Manual STAT: {req.primary_finding} ({req.confidence_score*100:.1f}%)",
            dispatch.get("execution_status", "DISPATCHED"),
            datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        )
    except Exception:
        pass
    return dispatch


@app.get("/api/v1/dpo/list")
async def get_saved_dpo_pairs(q: str = "", page: int = 1, page_size: int = 8):
    rows, total = local_db.query_dpo_pairs(q=q, page=page, page_size=page_size)
    return {"pairs": rows, "items": rows, "total": total, "page": page, "page_size": page_size}


@app.delete("/api/v1/dpo/{pair_id}")
async def remove_dpo_pair(pair_id: str):
    return {"deleted": local_db.delete_dpo_pair(pair_id)}


@app.get("/api/v1/dpo/export-jsonl", response_class=PlainTextResponse)
async def export_dpo_jsonl():
    return PlainTextResponse(local_db.export_all_dpo_jsonl(), media_type="application/x-ndjson")


@app.get("/api/v1/n8n/logs")
async def get_saved_n8n_logs(q: str = "", status: str = "", page: int = 1, page_size: int = 10):
    rows, total = local_db.query_n8n_logs(q=q, page=page, page_size=page_size)
    return {"logs": rows, "items": rows, "total": total, "page": page, "page_size": page_size}


@app.delete("/api/v1/n8n/logs/{log_id}")
async def remove_n8n_log(log_id: int):
    return {"deleted": local_db.delete_n8n_log(log_id)}

