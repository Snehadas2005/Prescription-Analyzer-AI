import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import uuid
import re
import spacy
from typing import Dict, List, Any
from transformers import pipeline
import logging

class ExtractionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize OCR
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize transformer model for medical NER
        self.medical_ner = pipeline(
            "ner",
            model="clinical-ai/Clinical-Bert-NER",
            aggregation_strategy="simple"
        )
        
        # Medicine database (expand this)
        self.medicine_database = self._load_medicine_database()
        
    def _load_medicine_database(self):
        """Load common medicine names"""
        return {
            'paracetamol', 'ibuprofen', 'amoxicillin', 'azithromycin',
            'omeprazole', 'metformin', 'amlodipine', 'aspirin',
            'ciprofloxacin', 'cephalexin', 'doxycycline', 'cetirizine',
            'pantoprazole', 'ranitidine', 'montelukast', 'losartan'
        }
    
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
                        "success": False,
                        "prescription_id": f"RX-{uuid.uuid4().hex[:8].upper()}",
                        "patient": {"name": "", "age": "", "gender": ""},
                        "doctor": {"name": "", "specialization": "", "registration": ""},
                        "medicines": [],
                        "confidence_score": 0.0,
                        "raw_text": "",
                        "error": result.error,
                        "message": "Analysis failed"
                    }
                
                # Convert to the expected format with all required fields
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
                        "registration": result.doctor.registration_number or "",
                    },
                    "medicines": [
                        {
                            "name": med.name or "",
                            "dosage": med.dosage or "",
                            "frequency": med.frequency or "",
                            "timing": med.instructions or "",
                            "duration": med.duration or "",
                            "quantity": int(med.quantity) if med.quantity and med.quantity.isdigit() else 1,
                        }
                        for med in result.medicines
                    ],
                    "confidence_score": float(result.confidence_score),
                    "raw_text": result.raw_text or "",
                    "message": "Analysis completed successfully",
                    "error": ""
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for better OCR"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilate to make text more readable
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    
    def _extract_text(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR"""
        try:
            results = self.reader.readtext(image)
            
            # Combine all text
            text = ' '.join([detection[1] for detection in results])
            
            return text
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""
    
    def _extract_patient_info(self, text: str) -> Dict[str, str]:
        """Extract patient information"""
        patient_info = {
            "name": "",
            "age": "",
            "gender": ""
        }
        
        # Extract name (looking for patterns)
        name_patterns = [
            r"Patient\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Name\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Mr\.|Mrs\.|Ms\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info["name"] = match.group(1).strip()
                break
        
        # Extract age
        age_patterns = [
            r"Age\s*:?\s*(\d{1,3})",
            r"(\d{1,3})\s*(?:years?|yrs?|Y)",
            r"(\d{1,3})\s*/\s*[MF]"
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 < age < 120:
                    patient_info["age"] = str(age)
                    break
        
        # Extract gender
        gender_patterns = [
            r"Sex\s*:?\s*(Male|Female|M|F)",
            r"Gender\s*:?\s*(Male|Female|M|F)",
            r"\d+\s*/\s*([MF])"
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).upper()
                patient_info["gender"] = "Male" if gender in ['M', 'MALE'] else "Female"
                break
        
        return patient_info
    
    def _extract_doctor_info(self, text: str) -> Dict[str, str]:
        """Extract doctor information"""
        doctor_info = {
            "name": "",
            "specialization": "",
            "registration": ""
        }
        
        # Extract doctor name
        doctor_patterns = [
            r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Doctor\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in doctor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor_info["name"] = match.group(1).strip()
                break
        
        # Extract specialization
        specializations = [
            'Cardiologist', 'Dermatologist', 'Pediatrician', 
            'General Physician', 'Surgeon', 'Neurologist'
        ]
        
        for spec in specializations:
            if spec.lower() in text.lower():
                doctor_info["specialization"] = spec
                break
        
        # Extract registration number
        reg_pattern = r"Reg\.?\s*(?:No\.?)?\s*:?\s*([A-Z0-9]+)"
        match = re.search(reg_pattern, text, re.IGNORECASE)
        if match:
            doctor_info["registration"] = match.group(1)
        
        return doctor_info
    
    def _extract_medicines(self, text: str) -> List[Dict[str, Any]]:
        """Extract medicine information"""
        medicines = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            # Check if line contains medicine
            medicine_info = self._parse_medicine_line(line)
            if medicine_info:
                medicines.append(medicine_info)
        
        return medicines
    
    def _parse_medicine_line(self, line: str) -> Dict[str, Any]:
        """Parse a single medicine line"""
        # Check if line contains known medicine
        line_lower = line.lower()
        
        found_medicine = None
        for medicine in self.medicine_database:
            if medicine in line_lower:
                found_medicine = medicine.title()
                break
        
        if not found_medicine:
            # Try to find medicine-like words
            words = re.findall(r'\b[A-Z][a-z]+\b', line)
            if words and len(words[0]) > 4:
                found_medicine = words[0]
        
        if not found_medicine:
            return None
        
        medicine_info = {
            "name": found_medicine,
            "dosage": self._extract_dosage(line),
            "frequency": self._extract_frequency(line),
            "timing": self._extract_timing(line),
            "duration": self._extract_duration(line),
            "quantity": self._extract_quantity(line)
        }
        
        return medicine_info
    
    def _extract_dosage(self, text: str) -> str:
        """Extract dosage information"""
        patterns = [
            r'(\d+\.?\d*)\s*(mg|ml|g|mcg)',
            r'(\d+)\s*(?:tablet|cap|tab)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "As prescribed"
    
    def _extract_frequency(self, text: str) -> str:
        """Extract frequency information"""
        frequencies = {
            'once': 'Once daily',
            'twice': 'Twice daily',
            'thrice': 'Three times daily',
            '1-0-1': 'Morning and Night',
            '1-1-1': 'Three times daily',
            '0-0-1': 'Night only',
            '1-0-0': 'Morning only',
            'bd': 'Twice daily',
            'tid': 'Three times daily',
            'qid': 'Four times daily'
        }
        
        text_lower = text.lower()
        for key, value in frequencies.items():
            if key in text_lower:
                return value
        
        return "As directed"
    
    def _extract_timing(self, text: str) -> str:
        """Extract meal timing"""
        if re.search(r'before.*(?:food|meal)', text, re.IGNORECASE):
            return "Before meals"
        elif re.search(r'after.*(?:food|meal)', text, re.IGNORECASE):
            return "After meals"
        elif re.search(r'with.*(?:food|meal)', text, re.IGNORECASE):
            return "With meals"
        
        return "Anytime"
    
    def _extract_duration(self, text: str) -> str:
        """Extract treatment duration"""
        patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "As prescribed"
    
    def _extract_quantity(self, text: str) -> int:
        """Extract quantity/number of packages"""
        # Look for patterns like "x 2", "2 strips", etc.
        patterns = [
            r'[x×]\s*(\d+)',
            r'(\d+)\s*(?:strip|packet|box)',
            r'qty\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 1
    
    def _calculate_confidence(
        self,
        patient: Dict,
        doctor: Dict,
        medicines: List
    ) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        # Patient info confidence
        patient_score = 0
        if patient.get('name'):
            patient_score += 0.4
        if patient.get('age'):
            patient_score += 0.3
        if patient.get('gender'):
            patient_score += 0.3
        scores.append(patient_score)
        
        # Doctor info confidence
        doctor_score = 0
        if doctor.get('name'):
            doctor_score += 0.6
        if doctor.get('specialization'):
            doctor_score += 0.4
        scores.append(doctor_score)
        
        # Medicine confidence
        if medicines:
            medicine_score = min(len(medicines) * 0.2, 1.0)
            scores.append(medicine_score)
        else:
            scores.append(0.0)
        
        # Average of all scores
        return sum(scores) / len(scores)
    
    def _generate_id(self) -> str:
        """Generate unique prescription ID"""
        import uuid
        return f"RX-{uuid.uuid4().hex[:8].upper()}"