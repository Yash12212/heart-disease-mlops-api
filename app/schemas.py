"""Request and response models for the Heart Disease Prediction API."""

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """A single patient's clinical measurements.

    These are the 13 features the model was trained on. Each field uses the
    same units and encoding as the Kaggle 'Heart Disease Dataset'.
    """

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia / thal test result (0-3)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 52,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 125,
                    "chol": 212,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 168,
                    "exang": 0,
                    "oldpeak": 1.0,
                    "slope": 2,
                    "ca": 2,
                    "thal": 3,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """The model's prediction for one patient."""

    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str = Field(..., description="Human readable prediction")
    probability: float = Field(..., description="Model confidence in the predicted class")
