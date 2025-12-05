from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import tempfile
import uuid
import logging
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.prescription_analyzer import EnhancedPrescriptionAnalyzer
except ImportError:
    try:
        from prescription_analyzer import EnhancedPrescriptionAnalyzer
    except ImportError as e:
        print(f"Failed to import EnhancedPrescriptionAnalyzer: {e}")
        raise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Prescription Analyzer API",
    description="Advanced prescription analysis using OCR and NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: Optional[EnhancedPrescriptionAnalyzer] = None

class AnalysisResponse(BaseModel):
    success: bool = Field(default=False)
    prescription_id: str = Field(default="")
    patient: Dict[str, str] = Field(default_factory=dict)
    doctor: Dict[str, str] = Field(default_factory=dict) 
    medicines: List[Dict[str, Any]] = Field(default_factory=list)
    diagnosis: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0)
    message: str = Field(default="")
    error: str = Field(default="")
    raw_text: str = Field(default="")

    class Config:
        extra = "ignore"

class HealthResponse(BaseModel):
    status: str
    analyzer_ready: bool
    timestamp: str
    cohere_available: bool = False

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup"""
    global analyzer
    try:
        logger.info("Initializing Enhanced Prescription Analyzer...")
        
        cohere_api_key = os.getenv('COHERE_API_KEY')
        tesseract_path = os.getenv('TESSERACT_PATH')
        
        analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=cohere_api_key,
            tesseract_path=tesseract_path,
            force_api=False
        )
        
        logger.info("✅ Enhanced Prescription Analyzer initialized successfully")
        
        if hasattr(analyzer, 'co') and analyzer.co:
            logger.info("✅ Cohere API available - Advanced NLP analysis enabled")
        else:
            logger.warning("⚠️ No Cohere API key - Using pattern-based analysis")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {e}")
        raise RuntimeError(f"Analyzer initialization failed: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Prescription Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cohere_available = analyzer is not None and hasattr(analyzer, 'co') and analyzer.co is not None
    
    return HealthResponse(
        status="healthy" if analyzer is not None else "unhealthy",
        analyzer_ready=analyzer is not None,
        timestamp=datetime.now().isoformat(),
        cohere_available=cohere_available
    )

# FIXED: Added this endpoint without /api prefix
@app.post("/analyze-prescription", response_model=AnalysisResponse)
async def analyze_prescription(file: UploadFile = File(...)):
    """
    Analyze uploaded prescription image
    """
    if not analyzer:
        logger.error("Analyzer not initialized")
        raise HTTPException(
            status_code=503, 
            detail="Analyzer not initialized"
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type"
        )
    
    temp_file_path = None
    
    try:
        file_extension = '.jpg'
        if file.content_type:
            if 'png' in file.content_type:
                file_extension = '.png'
            elif 'tiff' in file.content_type:
                file_extension = '.tiff'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            file_size = len(content)
            
            logger.info(f"📄 Received: {file.filename}, Size: {file_size} bytes")
            
            if file_size > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")
            
            if file_size < 1024:
                raise HTTPException(status_code=400, detail="File too small")
            
            temp_file.write(content)
            temp_file.flush()
        
        logger.info(f"🔄 Processing: {file.filename}")
        
        result = analyzer.analyze_prescription(temp_file_path)
        
        if not result.success:
            logger.warning(f"⚠️ Analysis failed: {result.error}")
            return AnalysisResponse(
                success=False,
                error=result.error,
                message="Failed to analyze prescription"
            )
        
        json_result = analyzer.to_json(result)
        
        logger.info(f"✅ Analysis completed for {result.prescription_id}")
        logger.info(f"   Doctor: {result.doctor.name or 'N/A'}")
        logger.info(f"   Patient: {result.patient.name or 'N/A'}")
        logger.info(f"   Medicines: {len(result.medicines)}")
        
        return AnalysisResponse(**json_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        return AnalysisResponse(
            success=False,
            error=str(e),
            message="An unexpected error occurred"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    logger.info(f"🚀 Starting API on {HOST}:{PORT}")
    
    uvicorn.run(
        "main_fixed:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
