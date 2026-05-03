"""
main.py - FastAPI service for the Bank Marketing classifier
Author: Joshua Bell - OIM3641 (Spring 2026)

Loads the PyCaret pipeline saved by discovery.py ('best_pipeline.pkl') and
exposes a POST /predict endpoint. Send a JSON body with the same feature
columns the model was trained on; the response is the predicted label
('yes'/'no') plus a probability score when the model supports it.

Run:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs for the Swagger UI.
"""

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pycaret.classification import load_model, predict_model

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "best_pipeline"  # load_model() appends .pkl

app = FastAPI(
    title="Bank Marketing Term-Deposit Classifier",
    description=(
        "Predicts whether a banking client will subscribe to a term deposit, "
        "served from a PyCaret pipeline trained in discovery.py."
    ),
    version="1.0.0",
)

# Load the pipeline once at startup
model = load_model(os.path.join(MODEL_DIR, MODEL_NAME))


class Client(BaseModel):
    """One Bank Marketing record. Defaults match a representative client."""

    age: int = Field(41, description="Client age in years")
    job: str = Field("management", description="Job category")
    marital: str = Field("married", description="Marital status")
    education: str = Field("tertiary", description="Education level")
    default: str = Field("no", description="Has credit in default?")
    balance: float = Field(1500.0, description="Average yearly balance (EUR)")
    housing: str = Field("yes", description="Has housing loan?")
    loan: str = Field("no", description="Has personal loan?")
    contact: str = Field("cellular", description="Contact communication type")
    day_of_week: int = Field(15, description="Last contact day of the month")
    month: str = Field("may", description="Last contact month")
    duration: int = Field(250, description="Last contact duration in seconds")
    campaign: int = Field(2, description="Number of contacts in this campaign")
    pdays: int = Field(-1, description="Days since last contact (-1 = never)")
    previous: int = Field(0, description="Contacts before this campaign")
    poutcome: str = Field("unknown", description="Previous campaign outcome")


class Prediction(BaseModel):
    prediction: str = Field(..., description="Predicted label ('yes' or 'no')")
    score: Optional[float] = Field(
        None, description="Model probability for the predicted class, if available"
    )


@app.get("/")
def root():
    return {
        "service": "Bank Marketing Term-Deposit Classifier",
        "model": MODEL_NAME,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    """Score a single client. Send a JSON object with the feature fields."""
    try:
        row = pd.DataFrame([client.model_dump()])
        result = predict_model(model, data=row)

        # PyCaret 3.x returns prediction_label / prediction_score columns
        label = str(result["prediction_label"].iloc[0])
        score = (
            float(result["prediction_score"].iloc[0])
            if "prediction_score" in result.columns
            else None
        )
        return Prediction(prediction=label, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
