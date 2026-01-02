import sys
import os
from pathlib import Path
from typing import Any, Dict
import uuid
import tempfile
import logging

# Add backend to Python path
# Helper path to find keys if needed
backend_path = Path(__file__).parent.parent.parent.parent / "backend"

# Import the analyzer from the same directory
try:
    from .prescription_analyzer import EnhancedPrescriptionAnalyzer
    print("✓ Successfully imported EnhancedPrescriptionAnalyzer from local directory")
except (ImportError, ValueError):
    # Fallback for running as a standalone script
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
    print("✓ Successfully imported EnhancedPrescriptionAnalyzer")
except ImportError as e:
    print(f"❌ Failed to import prescription_analyzer: {e}")
    print(f"   Backend path: {backend_path_str}")
    print(f"   Path exists: {backend_path.exists()}")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        """Initialize the extraction service with the enhanced analyzer"""
        try:
            # Get Cohere API key
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
                except Exception as e:
                    logger.warning(f"Could not load from keys.py: {e}")
            
            # Initialize the enhanced prescription analyzer
            # CRITICAL: Set force_api=False to allow pattern matching fallback
            self.analyzer = EnhancedPrescriptionAnalyzer(
                cohere_api_key=cohere_api_key,
                force_api=False  # This is CRITICAL - allows fallback
            )
            logger.info("✓ ExtractionService initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ExtractionService: {e}")
            raise
    
    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract prescription information using the enhanced analyzer
        """
        temp_file_path = None
        
        try:
            # Save image bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name
            
            logger.info(f"📄 Processing image: {temp_file_path}")
            logger.info(f"📦 Image size: {len(image_bytes)} bytes ({len(image_bytes)/1024:.2f} KB)")
            
            # CRITICAL: Analyze the prescription
            logger.info("🔍 Starting prescription analysis...")
            result = self.analyzer.analyze_prescription(temp_file_path)
            
            # Log the result details for debugging
            logger.info(f"📊 Analysis Result:")
            logger.info(f"   Success: {result.success}")
            logger.info(f"   Prescription ID: {result.prescription_id}")
            logger.info(f"   Patient Name: '{result.patient.name}'")
            logger.info(f"   Doctor Name: '{result.doctor.name}'")
            logger.info(f"   Medicines Count: {len(result.medicines)}")
            logger.info(f"   Confidence: {result.confidence_score:.2%}")
            logger.info(f"   Raw Text Length: {len(result.raw_text)} chars")
            
            if result.raw_text:
                logger.info(f"   Raw Text Preview: {result.raw_text[:200]}...")
            else:
                logger.warning("   ⚠️ NO RAW TEXT EXTRACTED!")
            
            # Check if analysis was successful
            if not result.success:
                logger.error(f"❌ Analysis failed: {result.error}")
                return {
                    "success": False,
                    "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                    "patient": {"name": "", "age": "", "gender": ""},
                    "doctor": {"name": "", "specialization": "", "registration_number": ""},
                    "medicines": [],
                    "confidence_score": 0.0,
                    "raw_text": result.raw_text or "",
                    "error": result.error,
                    "message": "Analysis failed - check OCR installation"
                }
            
            # Check if we got any meaningful data
            has_data = (
                result.patient.name or 
                result.doctor.name or 
                len(result.medicines) > 0 or
                len(result.raw_text) > 50
            )
            
            if not has_data:
                logger.warning("⚠️ Analysis succeeded but extracted NO data!")
                logger.warning("   This likely means OCR is not working properly")
                logger.warning("   Check if EasyOCR/Tesseract are installed")
            
            # Convert to the expected format
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
                "diagnosis": result.diagnosis or [],
                "confidence_score": float(result.confidence_score),
                "raw_text": result.raw_text or "",
                "message": "Analysis completed successfully" if has_data else "Analysis completed but no data extracted - check OCR",
                "error": "" if has_data else "OCR may not be working - no text extracted"
            }
            
            logger.info(f"✅ Returning response with {len(result.medicines)} medicines")
            return response_data
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR during extraction: {e}", exc_info=True)
            
            # Return detailed error
            return {
                "success": False,
                "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                "patient": {"name": "", "age": "", "gender": ""},
                "doctor": {"name": "", "specialization": "", "registration_number": ""},
                "medicines": [],
                "confidence_score": 0.0,
                "raw_text": "",
                "error": f"Extraction failed: {str(e)}",
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