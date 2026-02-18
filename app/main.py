# app/main.py

from fastapi import FastAPI
from fastapi import HTTPException
from app.logger import logger
from app.model import DefectRiskModel
from app.schema import PredictionInput, PredictionOutput

app = FastAPI(title="Defect Risk AI Service")

# Load model once at startup
risk_model = DefectRiskModel()


@app.get("/")
def root():
    return {"message": "Defect Risk Prediction API is running."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        logger.info(f"Received input: {input_data.features}")

        result = risk_model.predict(input_data.features)

        logger.info(f"Prediction result: {result}")

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
