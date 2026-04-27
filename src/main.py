"""
FastAPI backend for the AI Job Exposure prediction tool.

Endpoints:
  GET  /health              → liveness check
  POST /api/predict/job-title → resolve title via Claude + O*NET, predict
  POST /api/predict/manual    → predict from manually supplied features

Run:
  cd src && uvicorn main:app --reload --port 8000
"""

import sys
import os

# Ensure src/ is on the Python path before any local imports
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pickle
import traceback
from pipeline import (
    predict_ai_job_exposure,
    predict_manual,
    compute_fallback_stats,
    FEATURE_COLUMNS,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from logistic_regression import LogisticRegression  # noqa: F401 — needed for pickle

# ── App ─────────────────────────────────────────────────
app = FastAPI(title="AI Job Exposure API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ── Startup: load model, scaler, dataset once ──────────
model = None
scaler = None
dataset = None
fallback_stats = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, dataset, fallback_stats

    base = os.path.join(os.path.dirname(__file__), "..")

    model_path = os.path.join(base, "models", "logistic_regression_model.pkl")
    scaler_path = os.path.join(base, "models", "scaler.pkl")
    data_path = os.path.join(base, "dataset", "transformed_data.csv")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dataset = pd.read_csv(data_path)
    fallback_stats = compute_fallback_stats(dataset)

    print(f"✓ Model loaded  — {model_path}")
    print(f"✓ Scaler loaded — {scaler_path}")
    print(f"✓ Dataset loaded — {len(dataset)} occupations")


# ── Request schemas ─────────────────────────────────────
class JobTitleRequest(BaseModel):
    job_title: str = Field(..., min_length=1, max_length=500)
    job_description: Optional[str] = Field(None, max_length=5000)


class ManualRequest(BaseModel):
    isBright: int = Field(..., ge=0, le=1)
    isGreen: int = Field(..., ge=0, le=1)
    JobZone: int = Field(..., ge=1, le=5)
    MedianSalary: float = Field(..., ge=0)
    pct_computer: float = Field(..., ge=0, le=100)
    pct_physical: float = Field(..., ge=0, le=100)
    pct_communication: float = Field(..., ge=0, le=100)
    pct_analyze: float = Field(..., ge=0, le=100)
    pct_manage: float = Field(..., ge=0, le=100)
    pct_creative: float = Field(..., ge=0, le=100)
    pct_textnative: float = Field(..., ge=0, le=100)


# ── Endpoints ───────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dataset_rows": len(dataset) if dataset is not None else 0,
    }


@app.post("/api/predict/job-title")
def predict_job_title(req: JobTitleRequest):
    try:
        result = predict_ai_job_exposure(
            job_title=req.job_title,
            model=model,
            scaler=scaler,
            dataset=dataset,
            fallback_stats=fallback_stats,
            job_description=req.job_description,
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Internal prediction error")

    if "error" in result:
        status = 400 if result["error"] == "INVALID_JOB_TITLE" else 422
        raise HTTPException(status_code=status, detail=result["error"])

    return result


@app.post("/api/predict/manual")
def predict_manual_endpoint(req: ManualRequest):
    # Convert pct values from 0–100 (frontend) to 0–1 (model)
    features = {
        "isBright": req.isBright,
        "isGreen": req.isGreen,
        "JobZone": req.JobZone,
        "MedianSalary": req.MedianSalary,
        "pct_computer": req.pct_computer / 100,
        "pct_physical": req.pct_physical / 100,
        "pct_communication": req.pct_communication / 100,
        "pct_analyze": req.pct_analyze / 100,
        "pct_manage": req.pct_manage / 100,
        "pct_creative": req.pct_creative / 100,
        "pct_textnative": req.pct_textnative / 100,
    }

    try:
        result = predict_manual(features, model, scaler)
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Internal prediction error")

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result
