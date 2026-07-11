# ============================================================
# 🥇Tanara AI FastAPI
# ASEAN Public Health Early-Warning API
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "healthpulse_final_model.pkl"
FEATURE_FILE = BASE_DIR / "healthpulse_model_features.pkl"
LABEL_FILE = BASE_DIR / "healthpulse_risk_label_mapping.pkl"


# ============================================================
# 2. GLOBAL MODEL OBJECTS
# ============================================================

model = None
model_features = None
risk_label_mapping = None


# ============================================================
# 3. LOAD MODEL WHEN API STARTS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global model_features
    global risk_label_mapping

    required_files = [
        MODEL_FILE,
        FEATURE_FILE,
        LABEL_FILE,
    ]

    missing_files = [
        str(file_path.name)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "The following required model files are missing: "
            + ", ".join(missing_files)
        )

    try:
        model = joblib.load(MODEL_FILE)
        model_features = joblib.load(FEATURE_FILE)
        risk_label_mapping = joblib.load(LABEL_FILE)

    except Exception as error:
        raise RuntimeError(
            f"🥇Tanara AI model could not be loaded: {error}"
        ) from error

    print("🥇Tanara AI:  model loaded successfully.")

    yield

    model = None
    model_features = None
    risk_label_mapping = None


# ============================================================
# 4. INITIALIZE FASTAPI
# ============================================================

app = FastAPI(
    title="🥇Tanara AI API",
    description=(
        "An API for predicting the following year's public-health "
        "risk level across ASEAN countries using mortality, disease, "
        "nutrition, immunization, healthcare investment and workforce data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 5. CORS CONFIGURATION
# ============================================================
# This allows a Streamlit frontend to communicate with the API.
# For production, replace '*' with the actual frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# 6. REQUEST MODEL
# ============================================================

class HealthIndicators(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "country": "Cambodia",
                "year": 2025,
                "infant_mortality_rate": 18.5,
                "under_5_mortality_rate": 22.8,
                "maternal_mortality_rate": 145.0,
                "malaria_prevalence": 3.2,
                "tb_prevalence": 278.0,
                "undernourished_population": 14.6,
                "government_health_expenditure": 72.5,
                "dpt_immunization": 91.0,
                "measles_immunization": 88.0,
                "nurses_midwives_density": 1.1,
                "physicians_density": 0.3,
            }
        }
    )

    country: str = Field(
        ...,
        min_length=2,
        description="ASEAN country being assessed",
    )

    year: int = Field(
        ...,
        ge=2004,
        le=2100,
        description="Year of the health indicators",
    )

    infant_mortality_rate: float = Field(
        ...,
        ge=0,
        description="Infant deaths per 1,000 live births",
    )

    under_5_mortality_rate: float = Field(
        ...,
        ge=0,
        description="Under-five deaths per 1,000 live births",
    )

    maternal_mortality_rate: float = Field(
        ...,
        ge=0,
        description="Maternal deaths per 100,000 live births",
    )

    malaria_prevalence: float = Field(
        ...,
        ge=0,
        description="Malaria burden indicator used by the trained model",
    )

    tb_prevalence: float = Field(
        ...,
        ge=0,
        description="Tuberculosis burden indicator used by the trained model",
    )

    undernourished_population: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of the population that is undernourished",
    )

    government_health_expenditure: float = Field(
        ...,
        ge=0,
        description="Government health expenditure indicator",
    )

    dpt_immunization: float = Field(
        ...,
        ge=0,
        le=100,
        description="DPT immunization coverage percentage",
    )

    measles_immunization: float = Field(
        ...,
        ge=0,
        le=100,
        description="Measles immunization coverage percentage",
    )

    nurses_midwives_density: float = Field(
        ...,
        ge=0,
        description="Nurses and midwives density",
    )

    physicians_density: float = Field(
        ...,
        ge=0,
        description="Physician density",
    )


# ============================================================
# 7. RESPONSE MODEL
# ============================================================

class RiskProbability(BaseModel):
    low: float
    medium: float
    high: float


class PredictionResponse(BaseModel):
    country: str
    assessment_year: int
    forecast_year: int
    predicted_risk_level: str
    confidence: float
    probabilities: RiskProbability
    main_risk_signals: List[str]
    recommended_actions: List[str]
    message: str


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def decode_risk_label(encoded_class) -> str:
    """
    Convert the model's encoded output into Low, Medium or High.
    """

    possible_keys = [
        encoded_class,
        int(encoded_class)
        if isinstance(encoded_class, (int, float))
        else encoded_class,
        str(encoded_class),
    ]

    for key in possible_keys:
        if key in risk_label_mapping:
            return str(risk_label_mapping[key])

    fallback_mapping = {
        0: "Low",
        1: "Medium",
        2: "High",
        "0": "Low",
        "1": "Medium",
        "2": "High",
    }

    if encoded_class in fallback_mapping:
        return fallback_mapping[encoded_class]

    raise ValueError(
        f"Unknown model class returned: {encoded_class}"
    )


def identify_risk_signals(
    indicators: HealthIndicators,
) -> List[str]:
    """
    Identify prominent concern areas from the supplied indicators.

    These rules support explanation only.
    The machine-learning model still produces the final prediction.
    """

    signals = []

    if indicators.infant_mortality_rate >= 25:
        signals.append(
            "Infant mortality is elevated, indicating pressure on child-survival services."
        )

    if indicators.under_5_mortality_rate >= 30:
        signals.append(
            "Under-five mortality is elevated, indicating significant child-health vulnerability."
        )

    if indicators.maternal_mortality_rate >= 150:
        signals.append(
            "Maternal mortality is elevated, suggesting gaps in maternal and emergency obstetric care."
        )

    if indicators.malaria_prevalence >= 10:
        signals.append(
            "Malaria burden is elevated and may require stronger prevention and surveillance."
        )

    if indicators.tb_prevalence >= 200:
        signals.append(
            "Tuberculosis burden is elevated and requires stronger screening and treatment support."
        )

    if indicators.undernourished_population >= 15:
        signals.append(
            "Undernourishment is elevated, increasing population health vulnerability."
        )

    if indicators.dpt_immunization < 90:
        signals.append(
            "DPT immunization coverage is below the preferred prevention level."
        )

    if indicators.measles_immunization < 90:
        signals.append(
            "Measles immunization coverage is below the preferred prevention level."
        )

    if indicators.nurses_midwives_density < 2:
        signals.append(
            "Nurses and midwives density is low, limiting frontline healthcare capacity."
        )

    if indicators.physicians_density < 1:
        signals.append(
            "Physician density is low, indicating limited access to medical professionals."
        )

    if not signals:
        signals.append(
            "No major concern was detected using the current explanation thresholds."
        )

    return signals[:5]


def generate_recommendations(
    indicators: HealthIndicators,
) -> List[str]:
    actions = []

    if (
        indicators.infant_mortality_rate >= 25
        or indicators.under_5_mortality_rate >= 30
    ):
        actions.append(
            "Expand child-health services, early disease screening and immunization outreach."
        )

    if indicators.maternal_mortality_rate >= 150:
        actions.append(
            "Strengthen skilled birth attendance, emergency obstetric care and maternal referral systems."
        )

    if indicators.malaria_prevalence >= 10:
        actions.append(
            "Expand malaria surveillance, rapid testing, vector control and prevention campaigns."
        )

    if indicators.tb_prevalence >= 200:
        actions.append(
            "Increase TB screening, early case detection and treatment-adherence support."
        )

    if indicators.undernourished_population >= 15:
        actions.append(
            "Integrate nutrition support into maternal, child and community health programmes."
        )

    if (
        indicators.dpt_immunization < 90
        or indicators.measles_immunization < 90
    ):
        actions.append(
            "Conduct targeted vaccine catch-up campaigns and strengthen routine immunization."
        )

    if indicators.nurses_midwives_density < 2:
        actions.append(
            "Increase the deployment and retention of nurses and midwives in underserved areas."
        )

    if indicators.physicians_density < 1:
        actions.append(
            "Improve physician access through rural incentives, mobile clinics and telemedicine."
        )

    if not actions:
        actions.append(
            "Maintain current health investments and continue monitoring early-warning indicators."
        )

    return actions[:5]


# ============================================================
# 9. API ROUTES
# ============================================================

@app.get("/")
def home() -> Dict[str, str]:
    return {
        "name": "🥇VitaGuard AI API",
        "status": "running",
        "description": (
            "Predict next-year public-health risk using ASEAN "
            "health indicators."
        ),
        "documentation": "/docs",
    }


@app.get("/health")
def health_check() -> Dict[str, object]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": model_features is not None,
        "labels_loaded": risk_label_mapping is not None,
    }


@app.get("/model-info")
def model_information() -> Dict[str, object]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not currently available.",
        )

    return {
        "model_name": type(model).__name__,
        "target": "Following-year health risk level",
        "risk_classes": ["Low", "Medium", "High"],
        "number_of_features": len(model_features),
        "features": list(model_features),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_health_risk(
    indicators: HealthIndicators,
) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded.",
        )

    try:
        input_values = indicators.model_dump(
            exclude={"country", "year"}
        )

        input_df = pd.DataFrame(
            [[input_values[feature] for feature in model_features]],
            columns=model_features,
        )

        predicted_class = model.predict(input_df)[0]
        predicted_level = decode_risk_label(predicted_class)

        model_probabilities = model.predict_proba(input_df)[0]

        probability_dictionary = {
            decode_risk_label(class_code): float(probability)
            for class_code, probability in zip(
                model.classes_,
                model_probabilities,
            )
        }

        low_probability = probability_dictionary.get("Low", 0.0)
        medium_probability = probability_dictionary.get("Medium", 0.0)
        high_probability = probability_dictionary.get("High", 0.0)

        confidence = max(
            low_probability,
            medium_probability,
            high_probability,
        )

        signals = identify_risk_signals(indicators)
        recommendations = generate_recommendations(indicators)

        return PredictionResponse(
            country=indicators.country,
            assessment_year=indicators.year,
            forecast_year=indicators.year + 1,
            predicted_risk_level=predicted_level,
            confidence=round(confidence, 4),
            probabilities=RiskProbability(
                low=round(low_probability, 4),
                medium=round(medium_probability, 4),
                high=round(high_probability, 4),
            ),
            main_risk_signals=signals,
            recommended_actions=recommendations,
            message=(
                f"{indicators.country} is predicted to have a "
                f"{predicted_level} public-health risk level in "
                f"{indicators.year + 1}."
            ),
        )

    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The model expects a feature that was not supplied: "
                f"{error}"
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error