"""
Bidirectional n8n Workflow Orchestrator
Triggers automated emergency clinical triage, calendar booking, specialist notifications,
and continual DPO retraining pipelines when physician feedback crosses the threshold.
"""
import time
from typing import Dict, Any
import httpx


class N8nOrchestrator:
    def __init__(
        self,
        emergency_webhook_url: str = "http://localhost:5678/webhook/medvision-emergency-triage",
        retrain_webhook_url: str = "http://localhost:5678/webhook/medvision-auto-retrain",
    ):
        self.emergency_webhook_url = emergency_webhook_url
        self.retrain_webhook_url = retrain_webhook_url

    async def trigger_emergency_triage(
        self,
        patient: Dict[str, Any],
        vision_result: Dict[str, Any],
        clinical_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_ts = time.perf_counter()
        payload = {
            "event": "CRITICAL_RADIOLOGY_FINDING",
            "study_id": vision_result.get("study_id", ""),
            "patient_mrn": patient["mrn"],
            "patient_name": patient["full_name"],
            "age": patient.get("age", "N/A"),
            "vitals": patient.get("vitals", {}),
            "primary_finding": vision_result["primary_finding"],
            "confidence_score": vision_result["confidence_score"],
            "urgency": vision_result["urgency"],
            "clinical_risk_score": vision_result.get("clinical_risk_score", "N/A"),
            "ctr_ratio": vision_result.get("ctr_ratio", "N/A"),
            "roi_bbox": vision_result.get("roi_bounding_box", {"method": "Finding-Aware Grad-CAM++"}),
            "clinical_impression": clinical_report["generated_impression"],
            "recommended_actions": clinical_report["recommended_actions"],
            "actions_requested": [
                "GENERATE_DICOM_PDF_DOSSIER",
                "BOOK_STAT_PULMONOLOGY_SLOT",
                "DISPATCH_TELEGRAM_PAGER_ALERT",
                "CREATE_HL7_FHIR_EHR_ORDER",
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(self.emergency_webhook_url, json=payload)
                status = "SUCCESS" if resp.status_code < 300 else f"HTTP_{resp.status_code}"
        except Exception:
            # Graceful fallback in local dev when n8n container is offline
            status = "SIMULATED_QUEUED"

        elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
        return {
            "workflow_name": "MedVision_Emergency_Triage_v2",
            "trigger_event": "CRITICAL_RADIOLOGY_FINDING",
            "execution_status": status,
            "response_time_ms": elapsed_ms,
            "payload": payload,
        }

    async def trigger_auto_dpo_retraining(self, pending_pairs_count: int, threshold: int = 25) -> Dict[str, Any]:
        payload = {
            "event": "DPO_DATASET_THRESHOLD_REACHED",
            "pending_pairs_count": pending_pairs_count,
            "threshold": threshold,
            "target_gpu": "NVIDIA RTX 4080 Laptop (12GB VRAM)",
            "training_config": {
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "method": "QLoRA-DPO",
                "quantization": "4-bit NF4 (bitsandbytes)",
                "lora_r": 16,
                "lora_alpha": 32,
                "dpo_beta": 0.1,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(self.retrain_webhook_url, json=payload)
                status = "SUCCESS" if resp.status_code < 300 else f"HTTP_{resp.status_code}"
        except Exception:
            status = "SIMULATED_TRIGGERED"

        return {
            "workflow_name": "MedVision_Continual_DPO_Flywheel",
            "execution_status": status,
            "payload": payload,
        }
