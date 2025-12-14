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
        print(f"Failed to import: {e}")
        raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Prescription Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer: Optional[EnhancedPrescriptionAnalyzer] = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    try:
        logger.info("Initializing analyzer with TrOCR support...")
        
        analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=os.getenv('COHERE_API_KEY'),
            use_gpu=False,  # Set True if you have GPU
            force_api=False
        )
        
        logger.info("✅ Analyzer with hybrid OCR ready")
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        raise
    
@app.get("/")
async def root():
    return {
        "message": "AI Prescription Analyzer API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "cohere_available": hasattr(analyzer, 'co') and analyzer.co is not None if analyzer else False
    }

@app.post("/analyze-prescription")
async def analyze(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(503, "Analyzer not ready")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    
    temp_path = None
    try:
        ext = '.jpg'
        if 'png' in file.content_type: ext = '.png'
        elif 'tiff' in file.content_type: ext = '.tiff'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            temp_path = f.name
            content = await file.read()
            f.write(content)
            f.flush()
        
        logger.info(f"Processing {file.filename} ({len(content)} bytes)")
        
        result = analyzer.analyze_prescription(temp_path)
        
        if not result.success:
            logger.warning(f"Analysis failed: {result.error}")
            return JSONResponse({
                "success": False,
                "prescription_id": result.prescription_id or f"RX-{uuid.uuid4().hex[:8]}",
                "patient": {"name": "", "age": "", "gender": ""},
                "doctor": {"name": "", "specialization": "", "registration_number": ""},
                "medicines": [],
                "confidence_score": 0.0,
                "raw_text": result.raw_text or "",
                "error": result.error,
                "message": "Analysis failed"
            })
        
        json_data = analyzer.to_json(result)
        
        logger.info(f"✅ Success! ID: {result.prescription_id}")
        logger.info(f"   Doctor: {result.doctor.name}")
        logger.info(f"   Patient: {result.patient.name}")
        logger.info(f"   Meds: {len(result.medicines)}, Confidence: {result.confidence_score:.1%}")
        
        response = {
            "success": True,
            "prescription_id": json_data.get("prescription_id", result.prescription_id),
            "patient": {
                "name": json_data.get("patient", {}).get("name") or "",
                "age": json_data.get("patient", {}).get("age") or "",
                "gender": json_data.get("patient", {}).get("gender") or ""
            },
            "doctor": {
                "name": json_data.get("doctor", {}).get("name") or "",
                "specialization": json_data.get("doctor", {}).get("specialization") or "",
                "registration_number": json_data.get("doctor", {}).get("registration_number") or ""
            },
            "medicines": [
                {
                    "name": m.get("name") or "",
                    "dosage": m.get("dosage") or "",
                    "frequency": m.get("frequency") or "",
                    "timing": m.get("instructions") or m.get("timing") or "",
                    "duration": m.get("duration") or "",
                    "quantity": int(m.get("quantity", 1)) if str(m.get("quantity", "1")).isdigit() else 1,
                    "available": m.get("available", True)
                }
                for m in json_data.get("medicines", [])
            ],
            "confidence_score": float(json_data.get("confidence_score", result.confidence_score)),
            "raw_text": json_data.get("raw_text", result.raw_text) or "",
            "message": "Analysis completed",
            "error": ""
        }
        
        logger.info(f"📤 Response: confidence={response['confidence_score']:.2f}, meds={len(response['medicines'])}")
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return JSONResponse({
            "success": False,
            "prescription_id": f"RX-{uuid.uuid4().hex[:8]}",
            "patient": {"name": "", "age": "", "gender": ""},
            "doctor": {"name": "", "specialization": "", "registration_number": ""},
            "medicines": [],
            "confidence_score": 0.0,
            "raw_text": "",
            "error": str(e),
            "message": "Error occurred"
        })
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)