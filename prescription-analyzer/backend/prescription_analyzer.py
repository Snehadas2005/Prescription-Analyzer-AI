import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import re
import uuid
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    # Try common installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Tesseract found at: {path}")
            break
    else:
        print("⚠️ Tesseract not found in common paths. OCR may not work.")
        print("   Please set path manually or add to system PATH")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientInfo:
    name: str = ""
    age: str = ""
    gender: str = ""

@dataclass
class DoctorInfo:
    name: str = ""
    specialization: str = ""
    registration_number: str = ""

@dataclass
class MedicineInfo:
    name: str = ""
    dosage: str = ""
    frequency: str = ""
    instructions: str = ""
    duration: str = ""
    quantity: int = 1
    available: bool = True

@dataclass
class AnalysisResult:
    success: bool
    prescription_id: str
    patient: PatientInfo
    doctor: DoctorInfo
    medicines: List[MedicineInfo]
    confidence_score: float
    raw_text: str
    error: str = ""
    diagnosis: List[str] = None

    def __post_init__(self):
        if self.diagnosis is None:
            self.diagnosis = []

class EnhancedPrescriptionAnalyzer:
    def __init__(self, cohere_api_key: str = None, force_api: bool = False):
        """Initialize the analyzer"""
        logger.info("Initializing EnhancedPrescriptionAnalyzer...")
        
        # Initialize OCR readers
        try:
            logger.info("Loading EasyOCR reader (this may take a moment)...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✓ EasyOCR reader loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR: {e}")
            self.easyocr_reader = None
        
        logger.info("✓ EnhancedPrescriptionAnalyzer initialized")
    
    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """
        Main method to analyze prescription image
        """
        try:
            prescription_id = f"RX-{uuid.uuid4().hex[:8].upper()}"
            
            logger.info(f"Analyzing prescription image: {image_path}")
            
            # Step 1: Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return self._create_error_result(prescription_id, "Failed to read image file")
            
            logger.info(f"✓ Image loaded: {image.shape}")
            
            # Step 2: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            logger.info(f"✓ Converted to grayscale: {gray.shape}")
            
            # Step 3: Multiple preprocessing attempts
            processed_images = []
            
            # Method 1: Simple threshold
            try:
                _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(("Binary", binary1))
                logger.info("✓ Binary threshold preprocessing done")
            except Exception as e:
                logger.warning(f"Binary threshold failed: {e}")
            
            # Method 2: Adaptive threshold
            try:
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(("Adaptive", adaptive))
                logger.info("✓ Adaptive threshold preprocessing done")
            except Exception as e:
                logger.warning(f"Adaptive threshold failed: {e}")
            
            # Method 3: Original grayscale
            processed_images.append(("Grayscale", gray))
            
            # Step 4: Extract text using multiple methods
            all_texts = []
            best_confidence = 0.0
            best_text = ""
            
            for method_name, img in processed_images:
                logger.info(f"Trying OCR with {method_name} preprocessing...")
                
                # Try EasyOCR
                if self.easyocr_reader:
                    try:
                        logger.info(f"  Running EasyOCR on {method_name}...")
                        results = self.easyocr_reader.readtext(img, detail=1)
                        
                        if results:
                            text = " ".join([r[1] for r in results])
                            confidence = np.mean([r[2] for r in results])
                            
                            logger.info(f"  EasyOCR found {len(results)} text regions")
                            logger.info(f"  Text length: {len(text)} chars, confidence: {confidence:.2f}")
                            
                            all_texts.append(text)
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_text = text
                    except Exception as e:
                        logger.warning(f"  EasyOCR failed on {method_name}: {e}")
                
                # Try Tesseract
                try:
                    logger.info(f"  Running Tesseract on {method_name}...")
                    tess_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                    
                    if tess_text and len(tess_text.strip()) > 10:
                        logger.info(f"  Tesseract extracted {len(tess_text)} chars")
                        all_texts.append(tess_text)
                        
                        if len(tess_text) > len(best_text):
                            best_text = tess_text
                except Exception as e:
                    logger.warning(f"  Tesseract failed on {method_name}: {e}")
            
            # Combine all extracted texts
            if all_texts:
                combined_text = "\n".join(all_texts)
                logger.info(f"✓ Total text extracted: {len(combined_text)} chars")
                logger.info(f"  Preview: {combined_text[:200]}...")
            else:
                combined_text = ""
                logger.error("❌ NO TEXT EXTRACTED by any OCR method!")
            
            # Step 5: Parse the extracted text
            if not combined_text.strip():
                logger.warning("No text extracted, returning empty result")
                return AnalysisResult(
                    success=True,
                    prescription_id=prescription_id,
                    patient=PatientInfo(),
                    doctor=DoctorInfo(),
                    medicines=[],
                    confidence_score=0.0,
                    raw_text="",
                    error="No text could be extracted from image"
                )
            
            # Extract information
            patient = self._extract_patient_info(combined_text)
            doctor = self._extract_doctor_info(combined_text)
            medicines = self._extract_medicines(combined_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                best_confidence if best_confidence > 0 else 0.5,
                patient, doctor, medicines
            )
            
            logger.info(f"✓ Analysis complete:")
            logger.info(f"  Patient: {patient.name}")
            logger.info(f"  Doctor: {doctor.name}")
            logger.info(f"  Medicines: {len(medicines)}")
            logger.info(f"  Confidence: {confidence:.2%}")
            
            return AnalysisResult(
                success=True,
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                confidence_score=confidence,
                raw_text=combined_text,
                error=""
            )
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            return self._create_error_result(
                f"RX-{uuid.uuid4().hex[:8].upper()}", 
                str(e)
            )
    
    def _extract_patient_info(self, text: str) -> PatientInfo:
        """Extract patient information from text"""
        patient = PatientInfo()
        
        # Extract age - MORE FLEXIBLE PATTERNS
        age_patterns = [
            r'Age[:\s]+(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|yrs?|Y)',
            r'(\d{1,3})\s*/\s*[MF]',
            r'(?:^|\s)(\d{2})\s*[-/]',  # Age at start like "47-"
            r'(?:^|\s)(\d{2})\s+[a-zA-Z]',  # Age before text like "47 years"
        ]
        
        for pattern in age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    age = int(match.group(1))
                    if 0 < age < 120:
                        patient.age = str(age)
                        logger.info(f"  Found age: {age}")
                        break
                except:
                    continue
            if patient.age:
                break
        
        # Extract gender - MORE PATTERNS
        gender_patterns = [
            r'\b(Male|M/|M\b)',
            r'\b(Female|F/|F\b)',
            r'\b(F|M)\s*[,/]',
            r'[/\s](M|F)[/\s]'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender_char = match.group(1)[0].upper()
                patient.gender = "Male" if gender_char == 'M' else "Female"
                logger.info(f"  Found gender: {patient.gender}")
                break
        
        # Extract name - MUCH MORE FLEXIBLE
        # Look for handwritten names that appear before age or after common prefixes
        name_patterns = [
            r'Patient[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # NEW: Look for capitalized words before age/date patterns
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+\d{2}[-/\s]',
            # NEW: Look for names in first few lines
            r'(?:^|\n)\s*([A-Z][a-z]{2,15})\s*[,.]?\s*(?:\d|$)',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name_candidate = match.group(1).strip()
                # Filter out common medical terms
                excluded = ['age', 'date', 'clinic', 'hospital', 'doctor', 'patient', 
                           'male', 'female', 'morning', 'evening', 'delhi', 'consultant']
                if (len(name_candidate) >= 3 and 
                    name_candidate.lower() not in excluded and
                    not re.search(r'\d', name_candidate)):
                    patient.name = name_candidate
                    logger.info(f"  Found patient name: {name_candidate}")
                    break
            if patient.name:
                break
        
        return patient
    
    def _extract_doctor_info(self, text: str) -> DoctorInfo:
        """Extract doctor information from text"""
        doctor = DoctorInfo()
        
        # Extract doctor name - IMPROVED PATTERNS
        doc_patterns = [
            r'Dr\.?\s*\(?\s*(?:Mrs?\.?|Miss|Ms\.?)?\s*\)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Doctor[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # NEW: Match names that appear with MBBS/MD
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:M\.?B\.?B\.?S|MD|DGO)',
        ]
        
        for pattern in doc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor.name = match.group(1).strip()
                logger.info(f"  Found doctor name: {doctor.name}")
                break
        
        # Extract specialization - MORE COMPREHENSIVE
        specializations = [
            'Obstetrician', 'Gynaecologist', 'Gynecologist', 'OBGYN',
            'Cardiologist', 'Dermatologist', 'Pediatrician', 
            'General Physician', 'Surgeon', 'Neurologist',
            'Consultant', 'Physician'
        ]
        
        for spec in specializations:
            if spec.lower() in text.lower():
                doctor.specialization = spec
                logger.info(f"  Found specialization: {spec}")
                break
        
        # Extract registration number - MORE PATTERNS
        reg_patterns = [
            r'Reg\.?\s*(?:No\.?)?\s*:?\s*([A-Z0-9]+)',
            r'(?:M\.?B\.?B\.?S\.?|MD|DGO)[,\s]+([A-Z0-9]+)',
        ]
        
        for pattern in reg_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor.registration_number = match.group(1)
                logger.info(f"  Found registration: {doctor.registration_number}")
                break
        
        return doctor
    
    def _extract_medicines(self, text: str) -> List[MedicineInfo]:
        """Extract medicines from text - ENHANCED for handwritten prescriptions"""
        medicines = []
        seen_medicines = set()
        
        logger.info("  Extracting medicines...")
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip short lines and headers
            if len(line) < 2:
                continue
            
            # Skip obvious non-medicine lines
            skip_keywords = ['patient', 'doctor', 'date', 'clinic', 'hospital', 
                           'morning', 'evening', 'delhi', 'consultant', 'obstetrician']
            if any(kw in line.lower() for kw in skip_keywords):
                continue
            
            # METHOD 1: Look for Tab/Cap markers
            if re.search(r'\b(Tab|Cap|Syp|Inj)[.\s]', line, re.IGNORECASE):
                # Extract medicine name after Tab/Cap
                med_match = re.search(r'(?:Tab|Cap|Syp|Inj)[.\s]+([A-Z][a-z]+(?:[A-Z][a-z]*)?)', 
                                     line, re.IGNORECASE)
                if med_match:
                    med_name = med_match.group(1).strip()
                    
                    # Skip if too short or already found
                    if len(med_name) < 3 or med_name.lower() in seen_medicines:
                        continue
                    
                    seen_medicines.add(med_name.lower())
                    
                    # Extract dosage
                    dosage = "As prescribed"
                    dosage_match = re.search(r'(\d+\s*(?:mg|ml|g|gm))', line, re.IGNORECASE)
                    if dosage_match:
                        dosage = dosage_match.group(1)
                    
                    # Extract frequency
                    frequency = self._extract_frequency(line)
                    duration = self._extract_duration(line)
                    instructions = self._extract_instructions(line)
                    
                    medicine = MedicineInfo(
                        name=med_name,
                        dosage=dosage,
                        frequency=frequency,
                        duration=duration,
                        instructions=instructions,
                        quantity=1,
                        available=True
                    )
                    
                    medicines.append(medicine)
                    logger.info(f"    Found medicine: {med_name} {dosage} {frequency}")
                    continue
            
            # METHOD 2: Look for capitalized words that might be medicine names
            # Common medicine patterns in Indian prescriptions
            potential_medicines = re.findall(r'\b([A-Z][a-z]{2,12}(?:[A-Z][a-z]*)?)\b', line)
            
            for potential_med in potential_medicines:
                # Skip common words
                excluded = ['Tab', 'Cap', 'Syp', 'Morning', 'Evening', 'Daily', 
                           'Once', 'Twice', 'After', 'Before', 'With', 'Pain',
                           'Age', 'Date', 'Clinic', 'New', 'Delhi', 'Near']
                
                if (potential_med not in excluded and 
                    len(potential_med) >= 4 and
                    potential_med.lower() not in seen_medicines):
                    
                    # Check if this line has dosage markers (strong indicator)
                    has_dosage = re.search(r'\d+\s*(?:mg|ml|g|gm|bd|od)', line, re.IGNORECASE)
                    
                    if has_dosage:
                        seen_medicines.add(potential_med.lower())
                        
                        # Extract details
                        dosage_match = re.search(r'(\d+\s*(?:mg|ml|g|gm))', line, re.IGNORECASE)
                        dosage = dosage_match.group(1) if dosage_match else "As prescribed"
                        
                        frequency = self._extract_frequency(line)
                        duration = self._extract_duration(line)
                        instructions = self._extract_instructions(line)
                        
                        medicine = MedicineInfo(
                            name=potential_med,
                            dosage=dosage,
                            frequency=frequency,
                            duration=duration,
                            instructions=instructions,
                            quantity=1,
                            available=True
                        )
                        
                        medicines.append(medicine)
                        logger.info(f"    Found medicine: {potential_med} {dosage} {frequency}")
        
        logger.info(f"  Total medicines found: {len(medicines)}")
        return medicines
    
    def _extract_frequency(self, text: str) -> str:
        """Extract medication frequency"""
        text_lower = text.lower()
        
        # Common frequency patterns
        if re.search(r'\b(bd|twice)\b', text_lower):
            return "Twice daily"
        elif re.search(r'\b(od|once)\b', text_lower):
            return "Once daily"
        elif re.search(r'\b(tid|thrice)\b', text_lower):
            return "Three times daily"
        elif re.search(r'\b(qid)\b', text_lower):
            return "Four times daily"
        elif re.search(r'\b(sos|prn)\b', text_lower):
            return "As needed"
        
        # Look for patterns like "1-0-1" or "1-1-1"
        pattern_match = re.search(r'(\d)-(\d)-(\d)', text)
        if pattern_match:
            morning, afternoon, night = pattern_match.groups()
            total = int(morning) + int(afternoon) + int(night)
            if total == 1:
                return "Once daily"
            elif total == 2:
                return "Twice daily"
            elif total == 3:
                return "Three times daily"
        
        return "As directed"
    
    def _extract_duration(self, text: str) -> str:
        """Extract medication duration"""
        duration_patterns = [
            r'(\d+)\s*(?:day|days|d\b)',
            r'(\d+)\s*(?:week|weeks|wk|wks)',
            r'(\d+)\s*(?:month|months|mo)',
            r'x\s*(\d+)',  # Like "x 2" meaning 2 packages/duration
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num = match.group(1)
                if 'week' in pattern:
                    return f"{num} weeks"
                elif 'month' in pattern:
                    return f"{num} months"
                else:
                    return f"{num} days"
        
        return "As prescribed"
    
    def _extract_instructions(self, text: str) -> str:
        """Extract medication instructions"""
        text_lower = text.lower()
        instructions = []
        
        if 'after' in text_lower and ('food' in text_lower or 'meal' in text_lower):
            instructions.append("After meals")
        elif 'before' in text_lower and ('food' in text_lower or 'meal' in text_lower):
            instructions.append("Before meals")
        elif 'with' in text_lower and ('food' in text_lower or 'meal' in text_lower):
            instructions.append("With meals")
        
        if 'empty stomach' in text_lower:
            instructions.append("On empty stomach")
        
        if 'bedtime' in text_lower or 'night' in text_lower:
            instructions.append("At bedtime")
        
        return ", ".join(instructions) if instructions else ""
    
    def _calculate_confidence(self, ocr_confidence: float, patient: PatientInfo, 
                            doctor: DoctorInfo, medicines: List[MedicineInfo]) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        # OCR confidence
        scores.append(ocr_confidence * 0.3)
        
        # Patient info score
        patient_score = 0.0
        if patient.name: patient_score += 0.5
        if patient.age: patient_score += 0.25
        if patient.gender: patient_score += 0.25
        scores.append(patient_score * 0.2)
        
        # Doctor info score
        doctor_score = 0.0
        if doctor.name: doctor_score += 0.6
        if doctor.specialization: doctor_score += 0.2
        if doctor.registration_number: doctor_score += 0.2
        scores.append(doctor_score * 0.2)
        
        # Medicine score
        medicine_score = min(1.0, len(medicines) / 3.0)
        scores.append(medicine_score * 0.3)
        
        return sum(scores)
    
    def _create_error_result(self, prescription_id: str, error: str) -> AnalysisResult:
        """Create an error result"""
        return AnalysisResult(
            success=False,
            prescription_id=prescription_id,
            patient=PatientInfo(),
            doctor=DoctorInfo(),
            medicines=[],
            confidence_score=0.0,
            raw_text="",
            error=error
        )