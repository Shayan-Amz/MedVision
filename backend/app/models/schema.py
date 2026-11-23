from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class PatientCreate(BaseModel):
    mrn: str = Field(..., example="MRN-2026-8841")
    full_name: str = Field(..., example="Arthur Pendelton")
    age: int = Field(..., ge=0, le=130, example=58)
    sex: str = Field(..., example="Male")
    chief_complaint: str = Field(..., example="Acute dyspnea, pleuritic chest pain, fever 38.9C for 72h")
    clinical_history: Dict[str, Any] = Field(
        default_factory=lambda: {"comorbidities": ["Type 2 Diabetes", "Hypertension"], "smoking": "Former (15 pack-years)"}
    )
    vitals: Dict[str, Any] = Field(
        default_factory=lambda: {"spO2": "91%", "hr": 108, "bp": "142/88", "temp_c": 38.9, "rr": 24}
    )

class VisionPredictionOut(BaseModel):
    study_id: UUID
    model_version: str
    primary_finding: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    roi_bounding_box: Dict[str, float]
    gradcam_activation_stats: Dict[str, float]
    urgency: str
    inference_latency_ms: float

class ClinicalReportOut(BaseModel):
    report_id: UUID
    study_id: UUID
    llm_model_version: str
    generated_findings: str
    generated_impression: str
    differential_diagnosis: List[Dict[str, Any]]
    recommended_actions: List[str]
    faithfulness_score: float

class PhysicianFeedbackRequest(BaseModel):
    study_id: UUID
    report_id: UUID
    physician_name: str = Field(..., example="Dr. Elena Rostova, MD (Thoracic Radiology)")
    edited_report_text: str
    clinician_notes: Optional[str] = Field(None, example="Added explicit ruling out of tension pneumothorax and adjusted antibiotic renal dosing.")

class TrainingJobConfig(BaseModel):
    run_name: str = Field(..., example="qwen2.5-7b-clinical-dpo-v4")
    base_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    method: str = Field(default="QLoRA-DPO", pattern="^(QLoRA-SFT|QLoRA-DPO)$")
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=256)
    learning_rate: float = Field(default=5e-5)
    dpo_beta: float = Field(default=0.1, ge=0.01, le=0.5)
    num_epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=2, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=16)
