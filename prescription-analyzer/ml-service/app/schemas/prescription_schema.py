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
