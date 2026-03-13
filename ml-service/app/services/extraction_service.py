import os
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict

from .prescription_analyzer import PrescriptionAnalyzer

logger = logging.getLogger(__name__)


class ExtractionService:
    def __init__(self):
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            try:
                from .integration.keys import COHERE_API_KEY
                cohere_api_key = COHERE_API_KEY
                logger.info("✓ Cohere API key loaded from integration/keys.py")
            except ImportError:
                pass

        self.analyzer = PrescriptionAnalyzer(cohere_api_key=cohere_api_key)
        logger.info("✓ ExtractionService (stateless) initialised")

    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyse image bytes and return extracted prescription data. Nothing is stored."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                f.write(image_bytes)
                temp_path = f.name

            result = self.analyzer.analyze_prescription(temp_path)

            if not result.success:
                return {
                    "success": False,
                    "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                    "patient": {"name": "", "age": "", "gender": ""},
                    "doctor": {"name": "", "specialization": "", "registration_number": ""},
                    "medicines": [],
                    "diagnosis": [],
                    "confidence_score": 0.0,
                    "raw_text": "",
                    "detected_language": "en",
                    "error": result.error,
                    "message": "Analysis failed",
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
                        "quantity": int(m.quantity) if str(m.quantity).isdigit() else 1,
                        "available": m.available,
                    }
                    for m in result.medicines
                ],
                "diagnosis": result.diagnosis or [],
                "confidence_score": float(result.confidence_score),
                "raw_text": result.raw_text or "",
                # Pass through detected language from Gemini response
                "detected_language": getattr(result, "detected_language", "en"),
                "message": "Analysis completed successfully",
                "error": "",
            }

        except Exception as e:
            logger.error(f"❌ Extraction error: {e}", exc_info=True)
            return {
                "success": False,
                "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                "patient": {"name": "", "age": "", "gender": ""},
                "doctor": {"name": "", "specialization": "", "registration_number": ""},
                "medicines": [],
                "diagnosis": [],
                "confidence_score": 0.0,
                "raw_text": "",
                "detected_language": "en",
                "error": str(e),
                "message": "Unexpected error during analysis",
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass