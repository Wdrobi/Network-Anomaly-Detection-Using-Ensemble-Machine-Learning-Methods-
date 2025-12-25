"""
FastAPI Prediction Service
Provides a simple REST endpoint to run anomaly detection on new data.

Notes:
- Uses the trained Isolation Forest model saved at models/isolation_forest_model.pkl
- Reuses the DataPreprocessor to encode/scale incoming samples consistently with training
- Requires NSL-KDD train/test CSV files in data/KDDTrain+.csv and data/KDDTest+.csv to refit encoders/scaler

Run locally:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Example request (POST /predict):
{
  "records": [
    {
      "duration": 0,
      "protocol_type": "tcp",
      "service": "http",
      "flag": "SF",
      "src_bytes": 181,
      "dst_bytes": 5450,
      "land": 0,
      "wrong_fragment": 0,
      "urgent": 0,
      "hot": 0,
      "num_failed_logins": 0,
      "logged_in": 1,
      "num_compromised": 0,
      "root_shell": 0,
      "su_attempted": 0,
      "num_root": 0,
      "num_file_creations": 0,
      "num_shells": 0,
      "num_access_files": 0,
      "num_outbound_cmds": 0,
      "is_host_login": 0,
      "is_guest_login": 0,
      "count": 9,
      "srv_count": 9,
      "serror_rate": 0.00,
      "srv_serror_rate": 0.00,
      "rerror_rate": 0.00,
      "srv_rerror_rate": 0.00,
      "same_srv_rate": 1.00,
      "diff_srv_rate": 0.00,
      "srv_diff_host_rate": 0.00,
      "dst_host_count": 9,
      "dst_host_srv_count": 9,
      "dst_host_same_srv_rate": 1.00,
      "dst_host_diff_srv_rate": 0.00,
      "dst_host_same_src_port_rate": 1.00,
      "dst_host_srv_diff_host_rate": 0.00,
      "dst_host_serror_rate": 0.00,
      "dst_host_srv_serror_rate": 0.00,
      "dst_host_rerror_rate": 0.00,
      "dst_host_srv_rerror_rate": 0.00
    }
  ]
}
"""

import os
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import DataPreprocessor  # noqa: E402

app = FastAPI(title="Anomaly Detection API", version="1.0.0")

# Paths
TRAIN_PATH = os.getenv("NSL_KDD_TRAIN", "data/KDDTrain+.csv")
TEST_PATH = os.getenv("NSL_KDD_TEST", "data/KDDTest+.csv")
MODEL_PATH = os.getenv("IF_MODEL_PATH", "models/isolation_forest_model.pkl")

# Globals initialized at startup
preprocessor: DataPreprocessor | None = None
feature_order: List[str] = []
categorical_cols: List[str] = []
cat_modes: Dict[str, Any] = {}
numeric_medians: Dict[str, float] = {}
iso_model = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of samples to score")


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def _load_preprocessing() -> None:
    global preprocessor, feature_order, categorical_cols, cat_modes, numeric_medians

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Dataset files not found. Expected {TRAIN_PATH} and {TEST_PATH} for fitting encoders/scaler."
        )

    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_nsl_kdd_dataset(TRAIN_PATH, TEST_PATH)
    df_combined = pd.concat([train_df, test_df], ignore_index=True)

    # Feature schema
    feature_df = df_combined.drop(columns=["label"])
    feature_order = list(feature_df.columns)
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in feature_order if col not in categorical_cols]

    # Simple imputers (median for numeric, mode for categorical)
    numeric_medians = {col: float(feature_df[col].median()) if not feature_df[col].empty else 0.0 for col in numeric_cols}
    cat_modes = {col: str(feature_df[col].mode().iloc[0]) if not feature_df[col].empty else "" for col in categorical_cols}

    # Fit encoders and scaler using the existing preprocessing pipeline
    preprocessor.preprocess_pipeline(
        df_combined,
        target_col="label",
        categorical_cols=categorical_cols,
    )

    # Persist globals
    globals()["preprocessor"] = preprocessor
    globals()["feature_order"] = feature_order
    globals()["categorical_cols"] = categorical_cols
    globals()["cat_modes"] = cat_modes
    globals()["numeric_medians"] = numeric_medians


def _safe_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using fitted label encoders; unseen values fall back to mode."""
    assert preprocessor is not None

    for col in categorical_cols:
        encoder = preprocessor.label_encoders[col]
        known = set(encoder.classes_)
        fallback = cat_modes.get(col, encoder.classes_[0])
        mapped = df[col].astype(str).apply(lambda v: v if v in known else fallback)
        df[col] = encoder.transform(mapped)
    return df


def _preprocess_records(records: List[Dict[str, Any]]) -> np.ndarray:
    if preprocessor is None:
        raise RuntimeError("Preprocessor not initialized")

    if not records:
        raise ValueError("No records provided")

    df = pd.DataFrame(records)

    # Add missing columns, drop extras
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_order]

    # Impute missing values
    for col in feature_order:
        if col in categorical_cols:
            df[col] = df[col].fillna(cat_modes.get(col, ""))
        else:
            df[col] = df[col].fillna(numeric_medians.get(col, 0.0))

    # Encode categoricals and scale all features
    df_encoded = _safe_encode(df.copy())
    X_scaled = preprocessor.scaler.transform(df_encoded.values)
    return X_scaled


def _load_models() -> None:
    global iso_model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Isolation Forest model not found at {MODEL_PATH}. Run training pipeline first.")
    iso_model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event() -> None:
    _load_preprocessing()
    _load_models()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        X = _preprocess_records(req.records)
    except Exception as exc:  # pragma: no cover - runtime validation
        raise HTTPException(status_code=400, detail=str(exc))

    # Isolation Forest returns 1 for normal, -1 for anomaly
    raw_preds = iso_model.predict(X)
    anomaly_flags = (raw_preds == -1).astype(int)
    scores = -iso_model.decision_function(X)

    results = []
    for idx, (flag, score) in enumerate(zip(anomaly_flags, scores)):
        results.append({"index": idx, "anomaly": int(flag), "score": float(score)})

    return PredictResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
