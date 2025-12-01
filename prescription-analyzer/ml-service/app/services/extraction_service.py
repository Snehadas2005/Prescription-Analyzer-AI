from typing import Any, Dict, List
import uuid


class ExtractionService:
    async def extract(self, image_bytes: bytes) -> Dict[str, Any]:
        # TODO: replace with real OCR + NER logic
        # For now, return a dummy structure that matches what main.py expects

        return {
            "id": str(uuid.uuid4()),
            "patient": {
                "name": "Test Patient",
                "age": 30,
                "gender": "Other",
            },
            "doctor": {
                "name": "Dr. Demo",
                "registration_no": "ABC1234",
            },
            "medicines": [
                {
                    "name": "Paracetamol",
                    "dosage": "500mg",
                    "frequency": "1-0-1",
                    "duration": "5 days",
                    "instructions": "After food",
                }
            ],
            "confidence": 0.9,
        }
