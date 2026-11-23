-- MedVision & Clinical LLM Studio PostgreSQL Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TYPE urgency_level AS ENUM ('LOW', 'MODERATE', 'HIGH', 'CRITICAL');
CREATE TYPE imaging_modality AS ENUM ('CHEST_XRAY', 'BRAIN_MRI', 'DERM_LESION', 'CT_THORAX');
CREATE TYPE training_status AS ENUM ('QUEUED', 'RUNNING', 'EVALUATING', 'COMPLETED', 'FAILED');

CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mrn VARCHAR(32) UNIQUE NOT NULL,
    full_name VARCHAR(120) NOT NULL,
    age INT NOT NULL CHECK (age >= 0 AND age <= 130),
    sex VARCHAR(16) NOT NULL,
    chief_complaint TEXT NOT NULL,
    clinical_history JSONB DEFAULT '{}'::jsonb,
    vitals JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS imaging_studies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    study_code VARCHAR(40) UNIQUE NOT NULL,
    modality imaging_modality NOT NULL,
    image_uri TEXT NOT NULL,
    gradcam_uri TEXT,
    acquisition_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vision_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id UUID NOT NULL REFERENCES imaging_studies(id) ON DELETE CASCADE,
    model_version VARCHAR(64) NOT NULL,
    primary_finding VARCHAR(120) NOT NULL,
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    class_probabilities JSONB NOT NULL,
    roi_bounding_box JSONB NOT NULL,
    gradcam_activation_stats JSONB NOT NULL,
    urgency urgency_level NOT NULL,
    inference_latency_ms FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clinical_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id UUID NOT NULL REFERENCES imaging_studies(id) ON DELETE CASCADE,
    llm_model_version VARCHAR(80) NOT NULL,
    generated_findings TEXT NOT NULL,
    generated_impression TEXT NOT NULL,
    differential_diagnosis JSONB NOT NULL,
    recommended_actions JSONB NOT NULL,
    physician_edited_report TEXT,
    faithfulness_score FLOAT DEFAULT 0.95,
    is_approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(120),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dpo_feedback_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_id UUID NOT NULL REFERENCES imaging_studies(id) ON DELETE CASCADE,
    report_id UUID NOT NULL REFERENCES clinical_reports(id) ON DELETE CASCADE,
    system_prompt TEXT NOT NULL,
    clinical_context_prompt TEXT NOT NULL,
    rejected_response TEXT NOT NULL,
    chosen_response TEXT NOT NULL,
    edit_distance_ratio FLOAT NOT NULL,
    clinician_notes TEXT,
    used_in_training BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_name VARCHAR(120) UNIQUE NOT NULL,
    base_model VARCHAR(120) NOT NULL,
    method VARCHAR(32) NOT NULL, -- 'QLoRA-SFT' or 'QLoRA-DPO'
    hyperparameters JSONB NOT NULL,
    dataset_size INT NOT NULL,
    status training_status NOT NULL DEFAULT 'QUEUED',
    metrics_history JSONB DEFAULT '[]'::jsonb,
    final_eval_metrics JSONB DEFAULT '{}'::jsonb,
    vram_peak_gb FLOAT DEFAULT 0.0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS n8n_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(120) NOT NULL,
    trigger_event VARCHAR(80) NOT NULL,
    study_id UUID REFERENCES imaging_studies(id) ON DELETE SET NULL,
    payload JSONB NOT NULL,
    execution_status VARCHAR(32) NOT NULL,
    response_time_ms INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_studies_patient ON imaging_studies(patient_id);
CREATE INDEX idx_predictions_urgency ON vision_predictions(urgency);
CREATE INDEX idx_dpo_unused ON dpo_feedback_pairs(used_in_training) WHERE used_in_training = FALSE;
