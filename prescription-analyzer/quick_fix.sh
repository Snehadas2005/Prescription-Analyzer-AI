#!/bin/bash

# Quick Fix Script for ML Service
# This script helps you fix the ml-service files

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ML SERVICE QUICK FIX${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "ml-service" ]; then
    echo -e "${RED}❌ Error: ml-service directory not found${NC}"
    echo -e "${YELLOW}Please run this script from the prescription-analyzer root directory${NC}"
    exit 1
fi

echo -e "${YELLOW}This script will:${NC}"
echo -e "  1. Backup your current ML service files"
echo -e "  2. Update the extraction service"
echo -e "  3. Update the main.py file"
echo -e "  4. Ensure the backend prescription_analyzer.py is correct"
echo ""
echo -e "${YELLOW}Continue? (y/n)${NC}"
read -r response

if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create backup directory
BACKUP_DIR="ml-service-backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}Creating backups in $BACKUP_DIR...${NC}"

# Backup existing files
if [ -f "ml-service/app/services/extraction_service.py" ]; then
    cp ml-service/app/services/extraction_service.py "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Backed up extraction_service.py${NC}"
fi

if [ -f "ml-service/app/main.py" ]; then
    cp ml-service/app/main.py "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Backed up main.py${NC}"
fi

echo ""
echo -e "${BLUE}Step 1: Updating extraction_service.py${NC}"

cat > ml-service/app/services/extraction_service.py << 'EXTRACTION_SERVICE_EOF'
import sys
import os
from pathlib import Path
from typing import Any, Dict
import uuid
import tempfile
import logging

# Add the backend directory to Python path
backend_path = Path(__file__).parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from prescription_analyzer import EnhancedPrescriptionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        """Initialize the extraction service with the enhanced analyzer"""
        try:
            # Get Cohere API key from environment or integration keys
            cohere_api_key = os.getenv('COHERE_API_KEY')
            
            # Initialize the enhanced prescription analyzer
            self.analyzer = EnhancedPrescriptionAnalyzer(
                cohere_api_key=cohere_api_key,
                force_api=False  # Allow fallback to pattern matching
            )
            logger.info("✓ ExtractionService initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ExtractionService: {e}")
            raise
    
    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract prescription information using the enhanced analyzer
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            
        Returns:
            Dictionary containing extracted prescription information
        """
        temp_file_path = None
        
        try:
            # Save image bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name
            
            logger.info(f"📄 Processing image ({len(image_bytes)} bytes)")
            
            # Analyze the prescription using the backend analyzer
            result = self.analyzer.analyze_prescription(temp_file_path)
            
            # Check if analysis was successful
            if not result.success:
                logger.warning(f"⚠️ Analysis failed: {result.error}")
                return {
                    "success": False,
                    "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                    "patient": {"name": "", "age": "", "gender": ""},
                    "doctor": {"name": "", "specialization": "", "registration_number": ""},
                    "medicines": [],
                    "confidence_score": 0.0,
                    "raw_text": result.raw_text or "",
                    "error": result.error,
                    "message": "Analysis failed"
                }
            
            # Convert to the expected format with all required fields
            response_data = {
                "success": True,
                "prescription_id": result.prescription_id,
                "patient": {
                    "name": result.patient.name or "",
                    "age": result.patient.age or "",
                    "gender": result.patient.gender or "",
                },
                "doctor": {
                    "name": result.doctor.name or "",
                    "specialization": result.doctor.specialization or "",
                    "registration_number": result.doctor.registration_number or "",
                },
                "medicines": [
                    {
                        "name": med.name or "",
                        "dosage": med.dosage or "",
                        "frequency": med.frequency or "",
                        "timing": med.instructions or "",
                        "duration": med.duration or "",
                        "quantity": int(med.quantity) if med.quantity and str(med.quantity).isdigit() else 1,
                        "available": med.available
                    }
                    for med in result.medicines
                ],
                "confidence_score": float(result.confidence_score),
                "raw_text": result.raw_text or "",
                "message": "Analysis completed successfully",
                "error": ""
            }
            
            logger.info(f"✅ Analysis successful: {result.prescription_id}")
            logger.info(f"   Confidence: {result.confidence_score:.2%}")
            logger.info(f"   Patient: {result.patient.name or 'Not detected'}")
            logger.info(f"   Doctor: {result.doctor.name or 'Not detected'}")
            logger.info(f"   Medicines: {len(result.medicines)}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Error during extraction: {e}", exc_info=True)
            return {
                "success": False,
                "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                "patient": {"name": "", "age": "", "gender": ""},
                "doctor": {"name": "", "specialization": "", "registration_number": ""},
                "medicines": [],
                "confidence_score": 0.0,
                "raw_text": "",
                "error": str(e),
                "message": "An unexpected error occurred during analysis"
            }
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"🗑️ Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup temporary file: {e}")
EXTRACTION_SERVICE_EOF

echo -e "${GREEN}✓ Updated extraction_service.py${NC}"
echo ""

echo -e "${BLUE}Step 2: Updating main.py${NC}"

cat > ml-service/app/main.py << 'MAIN_PY_EOF'
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
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "analyzer_ready": extraction_service is not None
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
        raise HTTPException(400, "Invalid file type")
    
    try:
        image_bytes = await file.read()
        result = await extraction_service.extract(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
MAIN_PY_EOF

echo -e "${GREEN}✓ Updated main.py${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Fix Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}What was changed:${NC}"
echo -e "  ✓ ml-service/app/services/extraction_service.py - Updated to use backend analyzer"
echo -e "  ✓ ml-service/app/main.py - Simplified and fixed"
echo ""
echo -e "${YELLOW}Backups saved in: ${BACKUP_DIR}${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Make sure backend/prescription_analyzer.py exists and is correct"
echo -e "  2. Run: ${GREEN}cd backend && source venv312/bin/activate${NC}"
echo -e "  3. Run: ${GREEN}python main.py${NC}"
echo -e "  4. Test: ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo -e "${YELLOW}To test prescription analysis:${NC}"
echo -e "  ${GREEN}curl -X POST -F \"file=@prescription.jpg\" http://localhost:8000/analyze-prescription${NC}"
echo ""