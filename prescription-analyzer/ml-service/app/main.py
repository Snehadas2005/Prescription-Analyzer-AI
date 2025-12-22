from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.extraction_service import ExtractionService
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
    description="Machine Learning service for prescription analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extraction service
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
    return {
        "status": "healthy",
        "analyzer_ready": extraction_service is not None,
        "version": "1.0.0"
    }

@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    """Original endpoint"""
    return await analyze_prescription_internal(file)

@app.post("/analyze-prescription")
async def analyze_prescription(file: UploadFile = File(...)):
    """Go backend compatible endpoint"""
    return await analyze_prescription_internal(file)

async def analyze_prescription_internal(file: UploadFile):
    """Internal analysis function"""
    if not extraction_service:
        raise HTTPException(503, "Service not ready")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type. Please upload an image.")
    
    try:
        image_bytes = await file.read()
        
        # Validate file size
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(413, "File too large. Maximum size is 10MB.")
        
        result = await extraction_service.extract(image_bytes)
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    logger.info(f"🚀 Starting ML Service on {HOST}:{PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
