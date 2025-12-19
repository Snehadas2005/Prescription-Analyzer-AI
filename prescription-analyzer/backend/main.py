from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlservice.app.services.extraction_service import ExtractionService
from mlservice.app.schemas.prescription_schema import PrescriptionResponse
import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Prescription Analyzer - ML Service",
    description="Machine Learning service for prescription analysis with continuous learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize extraction service globally
extraction_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global extraction_service
    try:
        logger.info("🚀 Starting ML Service...")
        extraction_service = ExtractionService()
        logger.info("✅ ML Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML Service: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Prescription Analyzer - ML Service",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-prescription",
            "extract": "/extract",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-service",
        "analyzer_ready": extraction_service is not None,
        "version": "1.0.0"
    }

@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    """
    Extract information from prescription image (original endpoint)
    
    This endpoint maintains backward compatibility.
    """
    return await analyze_prescription_internal(file)

@app.post("/analyze-prescription")
async def analyze_prescription(file: UploadFile = File(...)):
    """
    Analyze prescription image (Go backend compatible endpoint)
    
    This endpoint is called by the Go backend service.
    Returns structured prescription data with patient, doctor, and medicine information.
    """
    return await analyze_prescription_internal(file)

async def analyze_prescription_internal(file: UploadFile):
    """
    Internal function for prescription analysis
    Used by both /extract and /analyze-prescription endpoints
    """
    if not extraction_service:
        logger.error("Extraction service not initialized")
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please try again in a moment."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        file_size = len(image_bytes)
        
        logger.info(f"📤 Received file: {file.filename}, Size: {file_size} bytes")
        
        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 10MB."
            )
        
        if file_size < 1024:
            raise HTTPException(
                status_code=400,
                detail="File too small. Please upload a valid prescription image."
            )
        
        # Extract information
        result = await extraction_service.extract(image_bytes)
        
        # Return the result
        if result.get("success"):
            logger.info(f"✅ Successfully analyzed prescription: {result.get('prescription_id')}")
            return JSONResponse(content=result)
        else:
            logger.warning(f"⚠️ Analysis completed with errors: {result.get('error')}")
            return JSONResponse(content=result, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/feedback")
async def receive_feedback(feedback_data: dict):
    """
    Receive feedback for continuous learning
    
    This endpoint stores user feedback to improve the model over time.
    """
    try:
        logger.info(f"📝 Received feedback for prescription: {feedback_data.get('prescription_id')}")
        
        # TODO: Store feedback in database
        # TODO: Trigger retraining if threshold is met
        
        return {
            "success": True,
            "message": "Feedback received successfully"
        }
    except Exception as e:
        logger.error(f"❌ Error storing feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to store feedback"
        )

@app.get("/stats")
async def get_statistics():
    """Get ML service statistics"""
    try:
        # TODO: Implement statistics tracking
        return {
            "total_analyzed": 0,
            "total_feedback": 0,
            "model_version": "1.0.0",
            "last_training": None
        }
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get statistics"
        )

if __name__ == "__main__":
    # Get configuration from environment
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    logger.info(f"🚀 Starting ML Service on {HOST}:{PORT}")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )