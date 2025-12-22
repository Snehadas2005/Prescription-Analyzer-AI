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
