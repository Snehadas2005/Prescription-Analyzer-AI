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

@app.post("/feedback")
async def provide_feedback(feedback: dict):
    """Endpoint for providing feedback/corrections for self-learning"""
    if not extraction_service:
        raise HTTPException(503, "Service not ready")
    
    try:
        # Update knowledge base with corrected data
        extraction_service.analyzer._update_knowledge_base(feedback)
        logger.info(f"✓ Feedback processed for prescription {feedback.get('prescription_id')}")
        return {"success": True, "message": "Feedback received and learned"}
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        raise HTTPException(500, f"Failed to process feedback: {str(e)}")

async def analyze_prescription_internal(file: UploadFile):
    """Internal analysis function"""
    if not extraction_service:
        raise HTTPException(503, "Service not ready")
    
    # FIXED: More lenient content type checking
    # Accept any image type or if content_type is not set
    valid_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp']
    
    # Log the received content type for debugging
    logger.info(f"📥 Received file: {file.filename}")
    logger.info(f"📋 Content-Type: {file.content_type}")
    
    # Check if it's a valid image based on extension if content_type is missing or invalid
    if file.content_type and not file.content_type.startswith('image/'):
        # Check file extension as fallback
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
        has_valid_extension = any(file.filename.lower().endswith(ext) for ext in valid_extensions)
        
        if not has_valid_extension:
            logger.error(f"❌ Invalid file type: {file.content_type}, filename: {file.filename}")
            raise HTTPException(
                400, 
                f"Invalid file type. Received: {file.content_type}. Expected image file (JPEG, PNG, etc.)"
            )
        else:
            logger.info(f"✓ Valid image extension detected: {file.filename}")
    
    try:
        image_bytes = await file.read()
        file_size = len(image_bytes)
        
        logger.info(f"📦 File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
        
        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(413, "File too large. Maximum size is 10MB.")
        
        if file_size < 100:
            raise HTTPException(400, "File too small. Please upload a valid image.")
        
        # Validate it's actually an image by checking magic bytes
        if not _is_valid_image(image_bytes):
            raise HTTPException(400, "Invalid image file. File may be corrupted.")
        
        logger.info("✓ File validation passed, starting analysis...")
        result = await extraction_service.extract(image_bytes)
        
        logger.info(f"✓ Analysis complete: success={result.get('success')}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")

def _is_valid_image(data: bytes) -> bool:
    """Check if file is a valid image by examining magic bytes"""
    if len(data) < 12:
        return False
    
    # Check for common image format signatures
    signatures = [
        b'\xFF\xD8\xFF',           # JPEG
        b'\x89PNG\r\n\x1a\n',      # PNG
        b'GIF87a',                 # GIF
        b'GIF89a',                 # GIF
        b'BM',                     # BMP
        b'II\x2A\x00',             # TIFF (little endian)
        b'MM\x00\x2A',             # TIFF (big endian)
        b'RIFF',                   # WebP (check for WEBP after RIFF)
    ]
    
    for sig in signatures:
        if data.startswith(sig):
            return True
    
    # Special check for WebP (RIFF....WEBP)
    if data.startswith(b'RIFF') and b'WEBP' in data[:20]:
        return True
    
    return False

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