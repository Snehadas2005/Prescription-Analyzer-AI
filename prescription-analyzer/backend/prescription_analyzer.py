import logging
import uuid
from typing import List, Optional

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientInfo:
    def __init__(self, name: str = "", age: str = "", gender: str = ""):
        self.name = name
        self.age = age
        self.gender = gender


class DoctorInfo:
    def __init__(self, name: str = "", specialization: str = "", registration_number: str = ""):
        self.name = name
        self.specialization = specialization
        self.registration_number = registration_number


class MedicineInfo:
    def __init__(
        self,
        name: str = "",
        dosage: str = "",
        frequency: str = "",
        instructions: str = "",
        duration: str = "",
        quantity: int = 1,
        available: bool = True,
    ):
        self.name = name
        self.dosage = dosage
        self.frequency = frequency
        self.instructions = instructions
        self.duration = duration
        self.quantity = quantity
        self.available = available


class AnalysisResult:
    def __init__(
        self,
        success: bool,
        prescription_id: str,
        patient: PatientInfo,
        doctor: DoctorInfo,
        medicines: List[MedicineInfo],
        confidence_score: float,
        raw_text: str = "",
        error: str = "",
    ):
        self.success = success
        self.prescription_id = prescription_id
        self.patient = patient
        self.doctor = doctor
        self.medicines = medicines
        self.confidence_score = confidence_score
        self.raw_text = raw_text
        self.error = error


# --------- MAIN ANALYZER ---------

class EnhancedPrescriptionAnalyzer:
    """
    Core prescription analyzer.
    This class MUST stay independent from FastAPI and services.
    """

    def __init__(self, cohere_api_key: Optional[str] = None, force_api: bool = False):
        self.cohere_api_key = cohere_api_key
        self.force_api = force_api
        logger.info("✓ EnhancedPrescriptionAnalyzer initialized")

    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """
        Analyze a prescription image and return structured results.
        """

        try:
            # TODO: plug OCR + NER here
            # For now this is a safe placeholder implementation

            logger.info(f"Analyzing prescription image: {image_path}")

            prescription_id = f"RX-{uuid.uuid4().hex[:8].upper()}"

            patient = PatientInfo(name="", age="", gender="")
            doctor = DoctorInfo(name="", specialization="", registration_number="")
            medicines: List[MedicineInfo] = []

            confidence_score = 0.0
            raw_text = ""

            return AnalysisResult(
                success=True,
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                confidence_score=confidence_score,
                raw_text=raw_text,
            )

        except Exception as e:
            logger.error(f"Analyzer failed: {e}", exc_info=True)

            return AnalysisResult(
                success=False,
                prescription_id=f"RX-{uuid.uuid4().hex[:8].upper()}",
                patient=PatientInfo(),
                doctor=DoctorInfo(),
                medicines=[],
                confidence_score=0.0,
                raw_text="",
                error=str(e),
            )
