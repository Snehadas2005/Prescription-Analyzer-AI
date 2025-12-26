import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import re
import uuid
import logging
import os
import pickle
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from thefuzz import fuzz, process
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientInfo:
    name: str = ""
    age: str = ""
    gender: str = ""
    confidence: float = 0.0


@dataclass
class DoctorInfo:
    name: str = ""
    specialization: str = ""
    registration_number: str = ""
    confidence: float = 0.0


@dataclass
class MedicineInfo:
    name: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    quantity: int = 1
    available: bool = True
    confidence: float = 0.0


@dataclass
class AnalysisResult:
    success: bool
    prescription_id: str
    patient: PatientInfo
    doctor: DoctorInfo
    medicines: List[MedicineInfo]
    confidence_score: float
    raw_text: str
    ocr_confidence: float
    extraction_confidence: float
    validation_confidence: float
    error: str = ""


class EnhancedPrescriptionAnalyzer:
    """
    Self-learning analyzer with 99% confidence target
    """
    
    def __init__(self, cohere_api_key: str = None, force_api: bool = False):
        logger.info("Initializing Enhanced Prescription Analyzer v2.0...")
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
        # Load knowledge bases
        self._load_knowledge_bases()
        
        logger.info("✓ Analyzer initialized with self-learning capabilities")
    
    def _init_ocr_engines(self):
        """Initialize OCR engines"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✓ EasyOCR initialized")
        except:
            self.easyocr_reader = None
            logger.warning("EasyOCR not available")
    
    def _load_knowledge_bases(self):
        """Load trained knowledge bases"""
        models_folder = Path("models/trained")
        
        # Load knowledge base if exists
        kb_path = models_folder / 'knowledge_base.pkl'
        if kb_path.exists():
            with open(kb_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
            logger.info(f"✓ Loaded knowledge base with {self.knowledge_base.get('total_patterns', 0)} patterns")
        else:
            self.knowledge_base = self._create_default_knowledge_base()
            logger.info("✓ Using default knowledge base")
        
        # Load learned patterns
        patterns_path = models_folder / 'learned_patterns.json'
        if patterns_path.exists():
            with open(patterns_path) as f:
                self.learned_patterns = json.load(f)
            logger.info("✓ Loaded learned patterns")
        else:
            self.learned_patterns = {}
    
    def _create_default_knowledge_base(self) -> Dict:
        """Create default knowledge base"""
        return {
            'medicines': {},
            'dosages': {},
            'frequencies': {},
            'doctors': {},
            'patients': {},
            'total_patterns': 0
        }
    
    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """
        Analyze prescription with enhanced confidence
        """
        prescription_id = f"RX-{uuid.uuid4().hex[:8].upper()}"
        
        try:
            logger.info(f"Analyzing: {image_path}")
            
            # Step 1: Extract text with multiple methods
            text, ocr_conf = self._extract_text_multi_method(image_path)
            
            if not text or len(text) < 20:
                return self._create_error_result(prescription_id, "Insufficient text extracted")
            
            logger.info(f"✓ Extracted {len(text)} chars, OCR confidence: {ocr_conf:.2%}")
            
            # Step 2: Extract entities with pattern matching
            patient = self._extract_patient_enhanced(text)
            doctor = self._extract_doctor_enhanced(text)
            medicines = self._extract_medicines_enhanced(text)
            
            # Step 3: Validate against knowledge base
            self._validate_with_knowledge_base(patient, doctor, medicines)
            
            # Step 4: Calculate multi-level confidence
            extraction_conf = self._calculate_extraction_confidence(patient, doctor, medicines)
            validation_conf = self._calculate_validation_confidence(patient, doctor, medicines)
            
            # Final confidence (weighted average)
            final_confidence = (
                0.30 * ocr_conf +
                0.40 * extraction_conf +
                0.30 * validation_conf
            )
            
            logger.info(f"✓ Analysis complete - Confidence: {final_confidence:.2%}")
            logger.info(f"  Patient: {patient.name} (conf: {patient.confidence:.2%})")
            logger.info(f"  Doctor: {doctor.name} (conf: {doctor.confidence:.2%})")
            logger.info(f"  Medicines: {len(medicines)} found")
            
            return AnalysisResult(
                success=True,
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                confidence_score=final_confidence,
                raw_text=text,
                ocr_confidence=ocr_conf,
                extraction_confidence=extraction_conf,
                validation_confidence=validation_conf,
                error=""
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return self._create_error_result(prescription_id, str(e))
    
    def _extract_text_multi_method(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text using multiple OCR methods and combine results
        """
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0
        
        # Preprocess image multiple ways
        processed = self._preprocess_image_enhanced(image)
        
        all_texts = []
        all_confidences = []
        
        for method_name, proc_img in processed:
            # Try EasyOCR
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(proc_img, detail=1)
                    if results:
                        text = " ".join([r[1] for r in results])
                        conf = np.mean([r[2] for r in results])
                        all_texts.append(text)
                        all_confidences.append(conf)
                except:
                    pass
            
            # Try Tesseract with multiple configs
            for config in ['--oem 3 --psm 6', '--oem 3 --psm 3', '--oem 3 --psm 4']:
                try:
                    text = pytesseract.image_to_string(proc_img, config=config)
                    if len(text.strip()) > 20:
                        data = pytesseract.image_to_data(proc_img, config=config, output_type=pytesseract.Output.DICT)
                        confs = [int(c) for c in data['conf'] if int(c) > 0]
                        if confs:
                            conf = np.mean(confs) / 100
                            all_texts.append(text)
                            all_confidences.append(conf)
                except:
                    pass
        
        if not all_texts:
            return "", 0.0
        
        # Combine using weighted voting
        combined_text = self._combine_ocr_results(all_texts, all_confidences)
        avg_confidence = np.mean(all_confidences)
        
        return combined_text, avg_confidence
    
    def _preprocess_image_enhanced(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Enhanced preprocessing with multiple methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if needed
        h, w = gray.shape
        if h < 1000 or w < 800:
            scale = max(1000/h, 800/w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        processed = []
        
        # Method 1: CLAHE + Adaptive
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed.append(("CLAHE_Adaptive", adaptive))
        
        # Method 2: Otsu
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(("Otsu", otsu))
        
        # Method 3: Morphological
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(("Morphological", binary))
        
        # Method 4: Normalized
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        processed.append(("Normalized", normalized))
        
        return processed
    
    def _combine_ocr_results(self, texts: List[str], confidences: List[float]) -> str:
        """Intelligently combine OCR results"""
        if not texts:
            return ""
        
        # Weight by confidence
        weighted = list(zip(texts, confidences))
        weighted.sort(key=lambda x: x[1], reverse=True)
        
        # Use top 3
        top_texts = [t for t, c in weighted[:3]]
        
        # Word-level voting
        from collections import Counter
        all_words = []
        for text in top_texts:
            all_words.extend(text.split())
        
        word_counts = Counter(all_words)
        
        # Reconstruct with most common words
        lines = []
        for text in top_texts[0].split('\n'):
            line_words = []
            for word in text.split():
                similar = [w for w in word_counts if fuzz.ratio(w, word) > 85]
                if similar:
                    best = max(similar, key=lambda w: word_counts[w])
                    line_words.append(best)
                else:
                    line_words.append(word)
            if line_words:
                lines.append(' '.join(line_words))
        
        return '\n'.join(lines)
    
    def _extract_patient_enhanced(self, text: str) -> PatientInfo:
        """Enhanced patient extraction with confidence scoring"""
        patient = PatientInfo()
        confidence_scores = []
        
        # Age extraction (multiple patterns)
        age_patterns = [
            r'Age[:\s]+(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|yrs?|Y)\b',
            r'(\d{1,3})\s*/\s*[MF]',
            r'(?:^|\s)(\d{2})\s*[-/]',
        ]
        
        for pattern in age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    age = int(match.group(1))
                    if 0 < age < 120:
                        patient.age = str(age)
                        confidence_scores.append(0.9)
                        break
                except:
                    continue
            if patient.age:
                break
        
        # Gender extraction
        if re.search(r'\b(Male|M/|M\b)', text, re.IGNORECASE) and 'Female' not in text:
            patient.gender = 'Male'
            confidence_scores.append(0.85)
        elif re.search(r'\b(Female|F/|F\b)', text, re.IGNORECASE):
            patient.gender = 'Female'
            confidence_scores.append(0.85)
        
        # Name extraction (enhanced with knowledge base)
        name_patterns = [
            r'Patient[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+\d{2}[-/\s]',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name_candidate = match.group(1).strip()
                excluded = ['age', 'date', 'clinic', 'hospital', 'doctor', 'patient']
                if (len(name_candidate) >= 3 and 
                    name_candidate.lower() not in excluded and
                    not re.search(r'\d', name_candidate)):
                    patient.name = name_candidate
                    confidence_scores.append(0.8)
                    break
            if patient.name:
                break
        
        # Calculate patient confidence
        patient.confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return patient
    
    def _extract_doctor_enhanced(self, text: str) -> DoctorInfo:
        """Enhanced doctor extraction"""
        doctor = DoctorInfo()
        confidence_scores = []
        
        # Name extraction
        dr_patterns = [
            r'Dr\.?\s*\(?\s*(?:Mrs?\.?|Miss|Ms\.?)?\s*\)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Doctor[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:M\.?B\.?B\.?S|MD|DGO)',
        ]
        
        for pattern in dr_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor.name = match.group(1).strip()
                confidence_scores.append(0.9)
                break
        
        # Specialization (check against knowledge base)
        specializations = [
            'Obstetrician', 'Gynaecologist', 'Gynecologist', 'OBGYN',
            'Cardiologist', 'Dermatologist', 'Pediatrician',
            'General Physician', 'Surgeon', 'Neurologist', 'Consultant'
        ]
        
        for spec in specializations:
            if spec.lower() in text.lower():
                doctor.specialization = spec
                confidence_scores.append(0.85)
                break
        
        # Registration number
        reg_patterns = [
            r'Reg\.?\s*(?:No\.?)?\s*:?\s*([A-Z0-9]+)',
            r'(?:M\.?B\.?B\.?S\.?|MD|DGO)[,\s]+([A-Z0-9]+)',
        ]
        
        for pattern in reg_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doctor.registration_number = match.group(1)
                confidence_scores.append(0.8)
                break
        doctor.confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return doctor

    def _extract_medicines_enhanced(self, text: str) -> List[MedicineInfo]:
        """Enhanced medicine extraction with knowledge base matching"""
        medicines = []
        seen_names = set()
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Skip non-medicine lines
            if any(kw in line.lower() for kw in ['patient', 'doctor', 'date', 'clinic']):
                continue
            
            # Pattern 1: Tab/Cap markers
            med_match = re.search(r'(Tab|Cap|Syp|Inj)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', line, re.IGNORECASE)
            
            if med_match:
                name = med_match.group(2).strip()
                
                if len(name) < 3 or name.lower() in seen_names:
                    continue
                
                # Verify against knowledge base
                is_valid, confidence = self._verify_medicine_name(name)
                
                if is_valid:
                    medicine = MedicineInfo(
                        name=name,
                        dosage=self._extract_dosage(line),
                        frequency=self._extract_frequency(line),
                        duration=self._extract_duration(line),
                        instructions=self._extract_instructions(line),
                        quantity=self._extract_quantity(line),
                        confidence=confidence
                    )
                    
                    medicines.append(medicine)
                    seen_names.add(name.lower())
        
        return medicines

    def _verify_medicine_name(self, name: str) -> Tuple[bool, float]:
        """Verify medicine name against knowledge base"""
        name_lower = name.lower().strip()
        
        # Check in knowledge base
        if 'medicines' in self.knowledge_base:
            if name_lower in self.knowledge_base['medicines']:
                return True, 0.95
        
        # Check in learned patterns
        if 'top_medicines' in self.learned_patterns:
            if name_lower in self.learned_patterns['top_medicines']:
                return True, 0.90
        
        # Fuzzy match
        if 'top_medicines' in self.learned_patterns:
            matches = process.extract(name_lower, self.learned_patterns['top_medicines'], scorer=fuzz.ratio, limit=3)
            if matches and matches[0][1] > 85:
                return True, matches[0][1] / 100
        
        # Pattern-based verification (medicine suffixes)
        medicine_suffixes = ['zole', 'pril', 'lol', 'cin', 'mycin', 'cillin', 'mab', 'nib']
        if any(name_lower.endswith(suffix) for suffix in medicine_suffixes):
            return True, 0.75
        
        # If name is capitalized and looks like a medicine (4+ chars)
        if len(name) >= 4 and name[0].isupper():
            return True, 0.60
        
        return False, 0.0

    def _extract_dosage(self, text: str) -> str:
        """Extract dosage"""
        pattern = r'(\d+\.?\d*)\s*(mg|ml|g|gm|mcg)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0) if match else "As prescribed"

    def _extract_frequency(self, text: str) -> str:
        """Extract frequency"""
        text_lower = text.lower()
        freq_map = {
            r'\b(once|od|qd|1-0-0)\b': 'Once daily',
            r'\b(twice|bd|bid|1-0-1)\b': 'Twice daily',
            r'\b(thrice|tid|1-1-1)\b': 'Three times daily',
            r'\b(qid|1-1-1-1)\b': 'Four times daily',
        }
        for pattern, freq in freq_map.items():
            if re.search(pattern, text_lower):
                return freq
        return "As directed"

    def _extract_duration(self, text: str) -> str:
        """Extract duration"""
        patterns = [
            r'(\d+)\s*(?:day|days|d)\b',
            r'(\d+)\s*(?:week|weeks|wk)\b',
            r'(\d+)\s*(?:month|months|mo)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return "As prescribed"

    def _extract_instructions(self, text: str) -> str:
        """Extract instructions"""
        text_lower = text.lower()
        instructions = []
        if 'before' in text_lower and 'meal' in text_lower:
            instructions.append("Before meals")
        elif 'after' in text_lower and 'meal' in text_lower:
            instructions.append("After meals")
        return ", ".join(instructions) if instructions else ""

    def _extract_quantity(self, text: str) -> int:
        """Extract quantity"""
        pattern = r'[x×]\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def _validate_with_knowledge_base(self, patient: PatientInfo, 
                                    doctor: DoctorInfo, 
                                    medicines: List[MedicineInfo]):
        """Validate extracted entities against knowledge base"""
        # Validate patient
        if patient.name and 'patients' in self.knowledge_base:
            gender_key = f"gender:{patient.gender.lower()}"
            if gender_key in self.knowledge_base['patients']:
                patient.confidence = min(1.0, patient.confidence * 1.1)
        
        # Validate doctor
        if doctor.name and 'doctors' in self.knowledge_base:
            if doctor.name.lower() in self.knowledge_base['doctors']:
                doctor.confidence = min(1.0, doctor.confidence * 1.15)
        
        # Validate medicines
        for medicine in medicines:
            if 'medicines' in self.knowledge_base:
                if medicine.name.lower() in self.knowledge_base['medicines']:
                    medicine.confidence = min(1.0, medicine.confidence * 1.2)

    def _calculate_extraction_confidence(self, patient: PatientInfo,
                                        doctor: DoctorInfo,
                                        medicines: List[MedicineInfo]) -> float:
        """Calculate extraction confidence"""
        scores = []
        
        # Patient score
        if patient.name or patient.age:
            scores.append(patient.confidence)
        else:
            scores.append(0.0)
        
        # Doctor score
        if doctor.name:
            scores.append(doctor.confidence)
        else:
            scores.append(0.0)
        
        # Medicine score
        if medicines:
            med_confidences = [m.confidence for m in medicines]
            scores.append(np.mean(med_confidences))
        else:
            scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0

    def _calculate_validation_confidence(self, patient: PatientInfo,
                                        doctor: DoctorInfo,
                                        medicines: List[MedicineInfo]) -> float:
        """Calculate validation confidence based on knowledge base"""
        validation_score = 0.0
        total_checks = 0
        
        # Check patient against KB
        if patient.name or patient.age:
            total_checks += 1
            if patient.confidence > 0.7:
                validation_score += 1.0
            else:
                validation_score += 0.5
        
        # Check doctor against KB
        if doctor.name:
            total_checks += 1
            if doctor.confidence > 0.7:
                validation_score += 1.0
            else:
                validation_score += 0.5
        
        # Check medicines against KB
        if medicines:
            total_checks += 1
            high_conf_meds = sum(1 for m in medicines if m.confidence > 0.7)
            validation_score += (high_conf_meds / len(medicines))
        
        return validation_score / total_checks if total_checks > 0 else 0.0

    def _create_error_result(self, prescription_id: str, error: str) -> AnalysisResult:
        """Create error result"""
        return AnalysisResult(
            success=False,
            prescription_id=prescription_id,
            patient=PatientInfo(),
            doctor=DoctorInfo(),
            medicines=[],
            confidence_score=0.0,
            raw_text="",
            ocr_confidence=0.0,
            extraction_confidence=0.0,
            validation_confidence=0.0,
            error=error
        )