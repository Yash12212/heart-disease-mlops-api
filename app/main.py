"""FastAPI service that serves the trained Heart Disease prediction model.

The service loads the scikit-learn pipeline that was trained on Google Colab
(see notebook/Heart_Disease_Model_Training.ipynb) and exposes it over HTTP.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import PatientData, PredictionResponse

# The model was saved next to the project root during training.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

# The exact feature order the model expects.
FEATURE_ORDER = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Load the model once when the application starts.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the likelihood of heart disease from clinical measurements.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Small welcome message so we know the API is reachable."""
    return {"message": "Heart Disease Prediction API. See /docs for usage."}


@app.get("/health")
def health_check():
    """Health check used by Docker and monitoring tools."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    """Make a prediction for a single patient."""
    # Build a one row DataFrame with the same column order used in training.
    features = pd.DataFrame([data.model_dump()])[FEATURE_ORDER]

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])

    return PredictionResponse(
        prediction=prediction,
        label="Disease" if prediction == 1 else "No disease",
        probability=round(probability, 4),
    )
