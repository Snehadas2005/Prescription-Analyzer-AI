from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PatientInfo(BaseModel):
    """Patient information schema"""
    name: str = Field(default="", description="Patient's full name")
    age: str = Field(default="", description="Patient's age")
    gender: str = Field(default="", description="Patient's gender")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "age": "35",
                "gender": "Male"
            }
        }


class DoctorInfo(BaseModel):
    """Doctor information schema"""
    name: str = Field(default="", description="Doctor's full name")
    specialization: str = Field(default="", description="Doctor's specialization")
    registration_number: str = Field(default="", description="Medical registration number")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Dr. Jane Smith",
                "specialization": "General Physician",
                "registration_number": "MED123456"
            }
        }


class MedicineInfo(BaseModel):
    """Medicine information schema"""
    name: str = Field(default="", description="Medicine name")
    dosage: str = Field(default="", description="Dosage amount")
    frequency: str = Field(default="", description="How often to take")
    timing: str = Field(default="", description="When to take (before/after meals)")
    duration: str = Field(default="", description="How long to take")
    quantity: int = Field(default=1, description="Number of packages/strips")
    available: bool = Field(default=True, description="Availability status")

    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        """Ensure quantity is positive"""
        if v < 1:
            return 1
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Paracetamol",
                "dosage": "500 mg",
                "frequency": "Three times daily",
                "timing": "After meals",
                "duration": "5 days",
                "quantity": 2,
                "available": True
            }
        }


class PrescriptionResponse(BaseModel):
    """Main prescription analysis response schema"""
    success: bool = Field(default=True, description="Whether the analysis was successful")
    prescription_id: str = Field(..., description="Unique prescription identifier")
    patient: PatientInfo = Field(..., description="Patient information")
    doctor: DoctorInfo = Field(..., description="Doctor information")
    medicines: List[MedicineInfo] = Field(default_factory=list, description="List of prescribed medicines")
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the analysis (0-1)"
    )
    raw_text: str = Field(default="", description="Raw extracted text from OCR")
    message: str = Field(default="", description="Success or error message")
    error: str = Field(default="", description="Error details if any")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of analysis"
    )

    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is between 0 and 1"""
        return max(0.0, min(1.0, v))

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prescription_id": "RX20241219123456abcd1234",
                "patient": {
                    "name": "John Doe",
                    "age": "35",
                    "gender": "Male"
                },
                "doctor": {
                    "name": "Dr. Jane Smith",
                    "specialization": "General Physician",
                    "registration_number": "MED123456"
                },
                "medicines": [
                    {
                        "name": "Paracetamol",
                        "dosage": "500 mg",
                        "frequency": "Three times daily",
                        "timing": "After meals",
                        "duration": "5 days",
                        "quantity": 2,
                        "available": True
                    }
                ],
                "confidence_score": 0.85,
                "raw_text": "Dr. Jane Smith...",
                "message": "Analysis completed successfully",
                "error": "",
                "timestamp": "2024-12-19T12:34:56.789Z"
            }
        }


class FeedbackRequest(BaseModel):
    """Feedback submission schema for continuous learning"""
    prescription_id: str = Field(..., description="ID of the prescription")
    feedback_type: str = Field(..., description="Type: 'correction', 'confirmation', 'rating'")
    corrections: Optional[Dict[str, Any]] = Field(None, description="Corrected data if any")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5")
    comments: Optional[str] = Field(None, description="Additional comments")

    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        """Ensure feedback type is valid"""
        valid_types = ['correction', 'confirmation', 'rating']
        if v not in valid_types:
            raise ValueError(f"feedback_type must be one of {valid_types}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "prescription_id": "RX20241219123456abcd1234",
                "feedback_type": "correction",
                "corrections": {
                    "patient": {
                        "name": "John Smith"
                    }
                },
                "rating": None,
                "comments": "Patient name was incorrect"
            }
        }


class FeedbackResponse(BaseModel):
    """Response after submitting feedback"""
    success: bool = Field(default=True)
    message: str = Field(default="Feedback received successfully")
    feedback_id: Optional[str] = Field(None, description="Unique feedback ID")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Feedback received successfully",
                "feedback_id": "FB20241219123456"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(default="healthy", description="Service health status")
    analyzer_ready: bool = Field(default=False, description="Is analyzer initialized")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Current timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "analyzer_ready": True,
                "version": "1.0.0",
                "timestamp": "2024-12-19T12:34:56.789Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid file type",
                "detail": "Please upload an image file (JPEG, PNG, etc.)",
                "timestamp": "2024-12-19T12:34:56.789Z"
            }
        }


class StatsResponse(BaseModel):
    """Statistics response schema"""
    total_analyzed: int = Field(default=0, description="Total prescriptions analyzed")
    total_feedback: int = Field(default=0, description="Total feedback received")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    model_version: str = Field(default="1.0.0", description="Current model version")
    last_training: Optional[str] = Field(None, description="Last training timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "total_analyzed": 1234,
                "total_feedback": 567,
                "average_confidence": 0.82,
                "model_version": "1.0.0",
                "last_training": "2024-12-15T10:00:00Z"
            }
        }