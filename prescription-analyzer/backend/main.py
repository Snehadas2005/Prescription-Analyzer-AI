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

# Add parent directory to path to import prescription_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced prescription analyzer from backend directory
try:
    from backend.prescription_analyzer import EnhancedPrescriptionAnalyzer
except ImportError:
    # Fallback for different directory structures
    try:
        from prescription_analyzer import EnhancedPrescriptionAnalyzer
    except ImportError as e:
        print(f"Failed to import EnhancedPrescriptionAnalyzer: {e}")
        print("Please ensure prescription_analyzer.py is accessible")
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

# Pydantic models
class AnalysisResponse(BaseModel):
    success: bool = Field(default=False)
    prescription_id: str = Field(default="")
    patient: Dict[str, str] = Field(default_factory=dict)
    doctor: Dict[str, str] = Field(default_factory=dict) 
    medicines: List[Dict[str, Any]] = Field(default_factory=list)
    diagnosis: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0)
    patient_name: str = Field(default="")
    patient_age: int = Field(default=0)
    patient_gender: str = Field(default="")
    doctor_name: str = Field(default="")
    doctor_license: str = Field(default="")
    message: str = Field(default="")
    error: str = Field(default="")
    raw_text: str = Field(default="")

    class Config:
        extra = "ignore"

class HealthResponse(BaseModel):
    status: str
    analyzer_ready: bool
    timestamp: str
    uptime: str = "N/A"
    cohere_available: bool = False

def ensure_safe_response_data(data: dict) -> dict:
    """Ensure all response data is safe for Pydantic validation"""
    safe_data = {}
    
    safe_data['success'] = data.get('success', False)
    safe_data['prescription_id'] = data.get('prescription_id') or ""
    safe_data['patient'] = data.get('patient') or {}
    safe_data['doctor'] = data.get('doctor') or {}
    safe_data['medicines'] = data.get('medicines') or []
    safe_data['diagnosis'] = data.get('diagnosis') or []
    safe_data['confidence_score'] = data.get('confidence_score', 0.0)
    safe_data['message'] = data.get('message') or ""
    safe_data['error'] = data.get('error') or ""
    safe_data['raw_text'] = data.get('raw_text') or ""
    
    # Legacy fields
    safe_data['patient_name'] = data.get('patient_name') or ""
    safe_data['patient_gender'] = data.get('patient_gender') or ""
    safe_data['doctor_name'] = data.get('doctor_name') or ""
    safe_data['doctor_license'] = data.get('doctor_license') or ""
    
    # Handle patient_age
    patient_age = data.get('patient_age', 0)
    try:
        safe_data['patient_age'] = int(patient_age) if patient_age is not None else 0
    except (ValueError, TypeError):
        safe_data['patient_age'] = 0
    
    # Ensure nested dictionaries are safe
    if isinstance(safe_data['patient'], dict):
        patient_dict = safe_data['patient']
        patient_dict['name'] = patient_dict.get('name') or ""
        patient_dict['age'] = patient_dict.get('age') or ""
        patient_dict['gender'] = patient_dict.get('gender') or ""
    
    if isinstance(safe_data['doctor'], dict):
        doctor_dict = safe_data['doctor']
        doctor_dict['name'] = doctor_dict.get('name') or ""
        doctor_dict['specialization'] = doctor_dict.get('specialization') or ""
        doctor_dict['registration_number'] = doctor_dict.get('registration_number') or ""
    
    # Ensure medicines list is safe
    if isinstance(safe_data['medicines'], list):
        safe_medicines = []
        for med in safe_data['medicines']:
            if isinstance(med, dict):
                safe_med = {
                    'name': med.get('name') or "",
                    'dosage': med.get('dosage') or "",
                    'quantity': med.get('quantity') or "",
                    'frequency': med.get('frequency') or "",
                    'duration': med.get('duration') or "",
                    'instructions': med.get('instructions') or "",
                    'available': med.get('available', True)
                }
                safe_medicines.append(safe_med)
        safe_data['medicines'] = safe_medicines
    
    return safe_data

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
            force_api=False  # Allow fallback to pattern matching
        )
        
        logger.info("Enhanced Prescription Analyzer initialized successfully")
        
        if hasattr(analyzer, 'co') and analyzer.co:
            logger.info("Cohere API available - Advanced NLP analysis enabled")
        else:
            logger.warning("No Cohere API key - Using pattern-based analysis")
            
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise RuntimeError(f"Analyzer initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Prescription Analyzer API")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Prescription Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze": "/api/analyze-prescription"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    cohere_available = analyzer is not None and hasattr(analyzer, 'co') and analyzer.co is not None
    
    return HealthResponse(
        status="healthy" if analyzer is not None else "unhealthy",
        analyzer_ready=analyzer is not None,
        timestamp=datetime.now().isoformat(),
        uptime="N/A",
        cohere_available=cohere_available
    )

@app.post("/api/analyze-prescription", response_model=AnalysisResponse)
async def analyze_prescription(file: UploadFile = File(...)):
    """
    Analyze uploaded prescription image
    
    Args:
        file: Uploaded image file (JPEG, PNG, TIFF)
        
    Returns:
        AnalysisResponse with extracted prescription data
    """
    if not analyzer:
        raise HTTPException(
            status_code=503, 
            detail="Analyzer not initialized. Please check server configuration."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file"
        )
    
    temp_file_path = None
    
    try:
        # Create temporary file
        file_extension = '.jpg'
        if file.content_type:
            if 'png' in file.content_type:
                file_extension = '.png'
            elif 'tiff' in file.content_type:
                file_extension = '.tiff'
            elif 'bmp' in file.content_type:
                file_extension = '.bmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            file_size = len(content)
            
            # Check file size
            if file_size > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
            
            if file_size < 1024:
                raise HTTPException(status_code=400, detail="File too small. Please ensure the image is readable")
            
            temp_file.write(content)
            temp_file.flush()
        
        logger.info(f"Processing prescription file: {file.filename}, Size: {file_size} bytes")
        
        # Analyze prescription
        result = analyzer.analyze_prescription(temp_file_path)
        
        if not result.success:
            logger.warning(f"Analysis failed for prescription {result.prescription_id}: {result.error}")
            safe_error_data = ensure_safe_response_data({
                'success': False,
                'error': result.error,
                'message': "Failed to analyze prescription. Please ensure the image is clear."
            })
            return AnalysisResponse(**safe_error_data)
        
        # Convert to JSON format
        json_result = analyzer.to_json(result)
        
        logger.info(f"Analysis completed successfully for prescription {result.prescription_id}")
        
        # Ensure response data is safe
        safe_data = ensure_safe_response_data(json_result)
        
        try:
            return AnalysisResponse(**safe_data)
        except Exception as validation_error:
            logger.error(f"Validation error: {validation_error}")
            fallback_data = ensure_safe_response_data({
                'success': result.success,
                'prescription_id': result.prescription_id,
                'message': 'Analysis completed but response formatting had issues',
                'confidence_score': result.confidence_score
            })
            return AnalysisResponse(**fallback_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing prescription: {str(e)}")
        safe_error_data = ensure_safe_response_data({
            'success': False,
            'error': str(e),
            'message': "An unexpected error occurred during analysis."
        })
        return AnalysisResponse(**safe_error_data)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "success": True,
        "statistics": {
            "analyzer_status": "ready" if analyzer else "not_initialized",
            "cohere_available": (hasattr(analyzer, 'co') and analyzer.co is not None) if analyzer else False,
            "timestamp": datetime.now().isoformat()
        }
    }

# Global exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred.",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting AI Prescription Analyzer API on {HOST}:{PORT}")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )