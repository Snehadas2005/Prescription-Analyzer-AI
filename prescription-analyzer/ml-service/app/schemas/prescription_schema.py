from typing import Any, Dict, List
from pydantic import BaseModel


class PrescriptionResponse(BaseModel):
    prescription_id: str
    patient: Dict[str, Any]
    doctor: Dict[str, Any]
    medicines: List[Dict[str, Any]]
    confidence_score: float
