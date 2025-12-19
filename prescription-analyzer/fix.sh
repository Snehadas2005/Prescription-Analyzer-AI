#!/bin/bash

# Complete Fix Script - Fixes Everything!
# This script creates all missing files and fixes the ML service

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║        ${BLUE}AI PRESCRIPTION ANALYZER - COMPLETE FIX${CYAN}              ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify we're in the right place
if [ ! -d "ml-service" ] && [ ! -d "backend" ]; then
    echo -e "${RED}❌ Error: Not in prescription-analyzer directory${NC}"
    echo -e "${YELLOW}Please run this from the prescription-analyzer root directory${NC}"
    exit 1
fi

echo -e "${YELLOW}This script will:${NC}"
echo -e "  1. ✨ Create all missing schema files"
echo -e "  2. 🔧 Fix the extraction service"
echo -e "  3. 📝 Fix the main.py FastAPI app"
echo -e "  4. 📦 Create all __init__.py files"
echo -e "  5. 🧪 Test the imports"
echo ""
echo -e "${YELLOW}Continue? (y/n)${NC}"
read -r response

if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create backup
BACKUP_DIR="backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}📦 Creating backup in $BACKUP_DIR...${NC}"

[ -f "ml-service/app/main.py" ] && cp ml-service/app/main.py "$BACKUP_DIR/"
[ -f "ml-service/app/services/extraction_service.py" ] && cp ml-service/app/services/extraction_service.py "$BACKUP_DIR/"
[ -f "ml-service/app/schemas/prescription_schema.py" ] && cp ml-service/app/schemas/prescription_schema.py "$BACKUP_DIR/"

echo -e "${GREEN}✓ Backup created${NC}"
echo ""

# ============================================================================
# Step 1: Create directory structure
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: Creating directory structure...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

mkdir -p ml-service/app/schemas
mkdir -p ml-service/app/services  
mkdir -p ml-service/app/models
mkdir -p backend/logs
mkdir -p logs

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# ============================================================================
# Step 2: Create prescription_schema.py
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Creating prescription_schema.py...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cat > ml-service/app/schemas/prescription_schema.py << 'SCHEMA_EOF'
"""
Pydantic schemas for prescription data validation
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PatientInfo(BaseModel):
    """Patient information"""
    name: str = Field(default="")
    age: str = Field(default="")
    gender: str = Field(default="")


class DoctorInfo(BaseModel):
    """Doctor information"""
    name: str = Field(default="")
    specialization: str = Field(default="")
    registration_number: str = Field(default="")


class MedicineInfo(BaseModel):
    """Medicine information"""
    name: str = Field(default="")
    dosage: str = Field(default="")
    frequency: str = Field(default="")
    timing: str = Field(default="")
    duration: str = Field(default="")
    quantity: int = Field(default=1)
    available: bool = Field(default=True)

    @validator('quantity')
    def quantity_positive(cls, v):
        return max(1, v)


class PrescriptionResponse(BaseModel):
    """Prescription analysis response"""
    success: bool = Field(default=True)
    prescription_id: str
    patient: PatientInfo
    doctor: DoctorInfo
    medicines: List[MedicineInfo] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_text: str = Field(default="")
    message: str = Field(default="")
    error: str = Field(default="")

    @validator('confidence_score')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))


class FeedbackRequest(BaseModel):
    """Feedback submission"""
    prescription_id: str
    feedback_type: str
    corrections: Optional[Dict[str, Any]] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    comments: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    analyzer_ready: bool = False
    version: str = "1.0.0"
SCHEMA_EOF

echo -e "${GREEN}✓ prescription_schema.py created${NC}"
echo ""

# ============================================================================
# Step 3: Create extraction_service.py
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Creating extraction_service.py...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cat > ml-service/app/services/extraction_service.py << 'EXTRACTION_EOF'
import sys
import os
from pathlib import Path
from typing import Any, Dict
import uuid
import tempfile
import logging

# Add backend to path
backend_path = Path(__file__).parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from prescription_analyzer import EnhancedPrescriptionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        try:
            cohere_api_key = os.getenv('COHERE_API_KEY')
            self.analyzer = EnhancedPrescriptionAnalyzer(
                cohere_api_key=cohere_api_key,
                force_api=False
            )
            logger.info("✓ ExtractionService initialized")
        except Exception as e:
            logger.error(f"❌ Init failed: {e}")
            raise
    
    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(image_bytes)
                temp_file_path = f.name
            
            result = self.analyzer.analyze_prescription(temp_file_path)
            
            if not result.success:
                return {
                    "success": False,
                    "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                    "patient": {"name": "", "age": "", "gender": ""},
                    "doctor": {"name": "", "specialization": "", "registration_number": ""},
                    "medicines": [],
                    "confidence_score": 0.0,
                    "raw_text": "",
                    "error": result.error,
                    "message": "Analysis failed"
                }
            
            return {
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
                        "name": m.name or "",
                        "dosage": m.dosage or "",
                        "frequency": m.frequency or "",
                        "timing": m.instructions or "",
                        "duration": m.duration or "",
                        "quantity": int(m.quantity) if m.quantity and str(m.quantity).isdigit() else 1,
                        "available": m.available
                    }
                    for m in result.medicines
                ],
                "confidence_score": float(result.confidence_score),
                "raw_text": result.raw_text or "",
                "message": "Analysis completed",
                "error": ""
            }
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return {
                "success": False,
                "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                "patient": {"name": "", "age": "", "gender": ""},
                "doctor": {"name": "", "specialization": "", "registration_number": ""},
                "medicines": [],
                "confidence_score": 0.0,
                "raw_text": "",
                "error": str(e),
                "message": "Error occurred"
            }
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
EXTRACTION_EOF

echo -e "${GREEN}✓ extraction_service.py created${NC}"
echo ""

# ============================================================================
# Step 4: Create main.py
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Creating main.py...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cat > ml-service/app/main.py << 'MAIN_EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.extraction_service import ExtractionService
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Prescription Analyzer - ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extraction_service = None

@app.on_event("startup")
async def startup_event():
    global extraction_service
    try:
        logger.info("🚀 Starting ML Service...")
        extraction_service = ExtractionService()
        logger.info("✅ ML Service ready")
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        raise

@app.get("/")
async def root():
    return {"service": "ML Service", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "analyzer_ready": extraction_service is not None}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    return await analyze_internal(file)

@app.post("/analyze-prescription")
async def analyze(file: UploadFile = File(...)):
    return await analyze_internal(file)

async def analyze_internal(file: UploadFile):
    if not extraction_service:
        raise HTTPException(503, "Service not ready")
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    try:
        image_bytes = await file.read()
        result = await extraction_service.extract(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
MAIN_EOF

echo -e "${GREEN}✓ main.py created${NC}"
echo ""

# ============================================================================
# Step 5: Create __init__.py files
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 5: Creating __init__.py files...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cat > ml-service/app/__init__.py << 'EOF'
"""ML Service Application"""
__version__ = "1.0.0"
EOF

cat > ml-service/app/schemas/__init__.py << 'EOF'
"""Schemas package"""
from .prescription_schema import PrescriptionResponse, HealthCheckResponse
__all__ = ["PrescriptionResponse", "HealthCheckResponse"]
EOF

cat > ml-service/app/services/__init__.py << 'EOF'
"""Services package"""
from .extraction_service import ExtractionService
__all__ = ["ExtractionService"]
EOF

cat > ml-service/app/models/__init__.py << 'EOF'
"""Models package"""
EOF

echo -e "${GREEN}✓ All __init__.py files created${NC}"
echo ""

# ============================================================================
# Step 6: Test imports
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 6: Testing imports...${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cd ml-service
python3 << 'TEST_EOF'
import sys
sys.path.insert(0, '.')
try:
    from app.schemas.prescription_schema import PrescriptionResponse
    print("✓ PrescriptionResponse import successful")
    from app.services.extraction_service import ExtractionService
    print("✓ ExtractionService import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
TEST_EOF

TEST_RESULT=$?
cd ..

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Import test passed${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║                    ${GREEN}✅ FIX COMPLETE!${CYAN}                          ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}📁 File Structure:${NC}"
echo "ml-service/app/"
echo "├── __init__.py          ✓"
echo "├── main.py              ✓"
echo "├── schemas/"
echo "│   ├── __init__.py      ✓"
echo "│   └── prescription_schema.py ✓"
echo "├── services/"
echo "│   ├── __init__.py      ✓"
echo "│   └── extraction_service.py  ✓"
echo "└── models/"
echo "    └── __init__.py      ✓"
echo ""

echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo ""
echo -e "1️⃣  ${BLUE}Activate virtual environment:${NC}"
echo -e "   ${GREEN}cd backend${NC}"
echo -e "   ${GREEN}source venv312/bin/activate${NC}"
echo ""
echo -e "2️⃣  ${BLUE}Start the ML service:${NC}"
echo -e "   ${GREEN}python main.py${NC}"
echo ""
echo -e "3️⃣  ${BLUE}Test it:${NC}"
echo -e "   ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo -e "4️⃣  ${BLUE}Analyze a prescription:${NC}"
echo -e "   ${GREEN}curl -X POST -F \"file=@prescription.jpg\" http://localhost:8000/analyze-prescription${NC}"
echo ""

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Your ML service is ready to use!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}💾 Backup saved in: ${BACKUP_DIR}${NC}"
echo ""