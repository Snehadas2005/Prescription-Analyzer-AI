from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.extraction_service import ExtractionService
from app.services.training_service import TrainingService
from app.models.continuous_learner import ContinuousLearner
from app.schemas.prescription_schema import PrescriptionResponse
import uvicorn

app = FastAPI(title="Prescription AI Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
extraction_service = ExtractionService()
training_service = TrainingService()
continuous_learner = ContinuousLearner()

@app.post("/extract", response_model=PrescriptionResponse)
async def extract_prescription(file: UploadFile = File(...)):
    """Extract information from prescription image"""
    try:
        # Read image
        image_bytes = await file.read()
        
        # Extract information
        result = await extraction_service.extract(image_bytes)
        
        return PrescriptionResponse(
            prescription_id=result["id"],
            patient=result["patient"],
            doctor=result["doctor"],
            medicines=result["medicines"],
            confidence_score=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def receive_feedback(feedback: dict):
    """Receive feedback for continuous learning"""
    try:
        # Store feedback
        await training_service.store_feedback(feedback)
        
        # Trigger learning if threshold reached
        if await training_service.should_retrain():
            await continuous_learner.trigger_training()
        
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def trigger_training():
    """Manually trigger model training"""
    try:
        result = await continuous_learner.train_model()
        return {"status": "success", "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": continuous_learner.get_version()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)