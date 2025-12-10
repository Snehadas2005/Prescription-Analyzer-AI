from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.extraction_service import ExtractionService
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

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": "v1.0.0",
        "service": "ml-service"
    }

@app.post("/extract", response_model=PrescriptionResponse)
async def extract_prescription(file: UploadFile = File(...)):
    """Extract information from prescription image (original endpoint)"""
    return await analyze_prescription_internal(file)

@app.post("/analyze-prescription", response_model=PrescriptionResponse)
async def analyze_prescription(file: UploadFile = File(...)):
    """Analyze prescription image (Go backend compatible endpoint)"""
    return await analyze_prescription_internal(file)

async def analyze_prescription_internal(file: UploadFile):
    """Internal function for both endpoints"""
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
