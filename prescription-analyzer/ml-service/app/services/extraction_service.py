import sys
import os
from pathlib import Path
from typing import Any, Dict
import uuid
import tempfile

# Add the backend directory to Python path
backend_path = Path(__file__).parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from prescription_analyzer import EnhancedPrescriptionAnalyzer

class ExtractionService:
    def __init__(self):
        # Initialize the enhanced prescription analyzer
        cohere_api_key = os.getenv('COHERE_API_KEY')
        self.analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=cohere_api_key,
            force_api=False  # Allow fallback to pattern matching
        )
    
    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract prescription information using the enhanced analyzer"""
        
        # Save image bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Analyze the prescription
            result = self.analyzer.analyze_prescription(temp_file_path)
            
            if not result.success:
                return {
                    "id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                    "patient": {"name": "", "age": "", "gender": ""},
                    "doctor": {"name": "", "specialization": "", "registration_number": ""},
                    "medicines": [],
                    "confidence": 0.0,
                    "raw_text": "",
                    "error": result.error
                }
            
            # Convert to the expected format
            return {
                "id": result.prescription_id,
                "patient": {
                    "name": result.patient.name,
                    "age": result.patient.age,
                    "gender": result.patient.gender,
                },
                "doctor": {
                    "name": result.doctor.name,
                    "specialization": result.doctor.specialization,
                    "registration_number": result.doctor.registration_number,
                },
                "medicines": [
                    {
                        "name": med.name,
                        "dosage": med.dosage,
                        "frequency": med.frequency,
                        "timing": med.instructions,
                        "duration": med.duration,
                        "quantity": int(med.quantity) if med.quantity else 1,
                    }
                    for med in result.medicines
                ],
                "confidence": result.confidence_score,
                "raw_text": result.raw_text,
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)