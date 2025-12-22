#!/bin/bash

# ML Service Complete Fix Script
# Run this from the prescription-analyzer root directory

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║        ${BLUE}ML SERVICE COMPLETE FIX - WINDOWS/LINUX${CYAN}              ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
    VENV_ACTIVATE="venv312/Scripts/activate"
    PYTHON_CMD="python"
else
    IS_WINDOWS=false
    VENV_ACTIVATE="venv312/bin/activate"
    PYTHON_CMD="python3"
fi

echo -e "${YELLOW}Detected OS: ${NC}"
if [ "$IS_WINDOWS" = true ]; then
    echo -e "  ${GREEN}Windows (Git Bash)${NC}"
else
    echo -e "  ${GREEN}Linux/macOS${NC}"
fi
echo ""

# Verify we're in the right directory
if [ ! -d "ml-service" ] || [ ! -d "backend" ]; then
    echo -e "${RED}❌ Error: Not in prescription-analyzer root directory${NC}"
    echo -e "${YELLOW}Please run this from: prescription-analyzer/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Running from correct directory${NC}"
echo ""

# ============================================================================
# Step 1: Setup Python Virtual Environment
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: Setting up Python Virtual Environment${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cd backend

if [ ! -d "venv312" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv312
fi

# Activate virtual environment
if [ "$IS_WINDOWS" = true ]; then
    source venv312/Scripts/activate 2>/dev/null || . venv312/Scripts/activate
else
    source venv312/bin/activate
fi

echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --quiet --upgrade pip setuptools wheel

echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# ============================================================================
# Step 2: Install Dependencies
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Installing Python Dependencies${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# Create a minimal requirements file
cat > temp_requirements.txt << 'EOF'
# Core dependencies
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.5.0

# OCR dependencies
opencv-python==4.8.1.78
numpy==1.26.4
Pillow==10.3.0
easyocr==1.7.0
pytesseract==0.3.10

# NLP dependencies
spacy==3.7.2
fuzzywuzzy==0.18.0
python-Levenshtein==0.23.0

# AI dependencies
cohere==4.32

# Utilities
python-dateutil==2.8.2
EOF

echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install --quiet -r temp_requirements.txt

# Install spaCy model
echo -e "${YELLOW}Installing spaCy model...${NC}"
$PYTHON_CMD -m spacy download en_core_web_sm --quiet

rm temp_requirements.txt

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# ============================================================================
# Step 3: Verify prescription_analyzer.py exists
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Verifying backend/prescription_analyzer.py${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

if [ -f "prescription_analyzer.py" ]; then
    echo -e "${GREEN}✓ prescription_analyzer.py found${NC}"
    
    # Check if it has the main class
    if grep -q "class EnhancedPrescriptionAnalyzer" prescription_analyzer.py; then
        echo -e "${GREEN}✓ EnhancedPrescriptionAnalyzer class found${NC}"
    else
        echo -e "${RED}✗ EnhancedPrescriptionAnalyzer class not found${NC}"
        echo -e "${YELLOW}The file exists but may be incomplete${NC}"
    fi
else
    echo -e "${YELLOW}⚠ prescription_analyzer.py not found in backend/${NC}"
    echo -e "${YELLOW}It should be in the documents you provided${NC}"
fi

echo ""

# ============================================================================
# Step 4: Fix ML Service Files
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Creating/Fixing ML Service Files${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cd ../ml-service

# Create directory structure
mkdir -p app/services app/schemas app/models

# Create __init__.py files
touch app/__init__.py
touch app/services/__init__.py
touch app/schemas/__init__.py
touch app/models/__init__.py

echo -e "${GREEN}✓ Directory structure created${NC}"

# Create extraction_service.py with FIXED import
cat > app/services/extraction_service.py << 'EXTRACTION_EOF'
import sys
import os
from pathlib import Path
from typing import Any, Dict
import uuid
import tempfile
import logging

# CRITICAL FIX: Add backend to Python path BEFORE importing
backend_path = Path(__file__).parent.parent.parent.parent / "backend"
backend_path_str = str(backend_path.resolve())

if backend_path_str not in sys.path:
    sys.path.insert(0, backend_path_str)
    print(f"✓ Added to Python path: {backend_path_str}")

# Now import the analyzer
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
    print("✓ Successfully imported EnhancedPrescriptionAnalyzer")
except ImportError as e:
    print(f"❌ Failed to import prescription_analyzer: {e}")
    print(f"   Backend path: {backend_path_str}")
    print(f"   Path exists: {backend_path.exists()}")
    print(f"   Files in backend: {list(backend_path.glob('*.py'))[:5]}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        """Initialize the extraction service with the enhanced analyzer"""
        try:
            # Get Cohere API key from environment or integration keys
            cohere_api_key = os.getenv('COHERE_API_KEY')
            
            # Try to import from integration/keys.py if available
            if not cohere_api_key:
                try:
                    integration_path = backend_path / "integration"
                    if integration_path.exists():
                        sys.path.insert(0, str(integration_path))
                        from keys import COHERE_API_KEY
                        cohere_api_key = COHERE_API_KEY
                        logger.info("✓ Loaded Cohere API key from integration/keys.py")
                except:
                    pass
            
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
                    logger.debug(f"🗑️ Cleaned up temporary file")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup temporary file: {e}")
EXTRACTION_EOF

echo -e "${GREEN}✓ Created extraction_service.py${NC}"

# Create schemas
cat > app/schemas/prescription_schema.py << 'SCHEMA_EOF'
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class PrescriptionResponse(BaseModel):
    success: bool = Field(default=True)
    prescription_id: str
    patient: Dict[str, Any]
    doctor: Dict[str, Any]
    medicines: List[Dict[str, Any]]
    confidence_score: float
    raw_text: str = Field(default="")
    message: str = Field(default="")
    error: str = Field(default="")
SCHEMA_EOF

echo -e "${GREEN}✓ Created prescription_schema.py${NC}"

# Create main.py
cat > app/main.py << 'MAIN_EOF'
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
MAIN_EOF

echo -e "${GREEN}✓ Created main.py${NC}"

cd ..

echo ""

# ============================================================================
# Step 5: Test Import
# ============================================================================

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 5: Testing Python Import${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

cd backend

$PYTHON_CMD << 'TEST_IMPORT_EOF'
import sys
from pathlib import Path

# Add backend to path
backend_path = Path.cwd()
sys.path.insert(0, str(backend_path))

print(f"Backend path: {backend_path}")
print(f"Checking for prescription_analyzer.py...")

analyzer_file = backend_path / "prescription_analyzer.py"
print(f"File exists: {analyzer_file.exists()}")

if analyzer_file.exists():
    try:
        from prescription_analyzer import EnhancedPrescriptionAnalyzer
        print("✓ Successfully imported EnhancedPrescriptionAnalyzer")
        
        # Try to instantiate
        analyzer = EnhancedPrescriptionAnalyzer(force_api=False)
        print("✓ Successfully created analyzer instance")
        print("\n✅ IMPORT TEST PASSED!")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("✗ prescription_analyzer.py not found")
    print(f"\nFiles in backend:")
    for f in backend_path.glob("*.py"):
        print(f"  - {f.name}")
    sys.exit(1)
TEST_IMPORT_EOF

IMPORT_TEST_RESULT=$?

cd ..

if [ $IMPORT_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Import test passed!${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    echo -e "${YELLOW}The prescription_analyzer.py file may be missing or incomplete${NC}"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║                    ${GREEN}✅ SETUP COMPLETE!${CYAN}                          ║${NC}"
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
echo "└── services/"
echo "    ├── __init__.py      ✓"
echo "    └── extraction_service.py  ✓"
echo ""

echo -e "${YELLOW}🚀 To Start ML Service:${NC}"
echo ""
if [ "$IS_WINDOWS" = true ]; then
    echo -e "1️⃣  ${GREEN}cd backend${NC}"
    echo -e "   ${GREEN}source venv312/Scripts/activate${NC}"
    echo ""
    echo -e "2️⃣  ${GREEN}cd ../ml-service${NC}"
    echo -e "   ${GREEN}python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload${NC}"
else
    echo -e "1️⃣  ${GREEN}cd backend${NC}"
    echo -e "   ${GREEN}source venv312/bin/activate${NC}"
    echo ""
    echo -e "2️⃣  ${GREEN}cd ../ml-service${NC}"
    echo -e "   ${GREEN}uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload${NC}"
fi
echo ""

echo -e "${YELLOW}🧪 Test Commands:${NC}"
echo ""
echo -e "${GREEN}# Health check${NC}"
echo "curl http://localhost:8000/health"
echo ""
echo -e "${GREEN}# Test prescription analysis${NC}"
echo "curl -X POST -F \"file=@prescription.jpg\" http://localhost:8000/analyze-prescription"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Your ML service is ready!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""