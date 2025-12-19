"""
FIXED VERSION - prescription_analyzer.py
Place this file at: backend/prescription_analyzer.py

This fixes:
1. ValueError: extract_text returning wrong number of values
2. OpenCV preprocessing failing on color images
"""

from __future__ import annotations
import base64
import cv2
import numpy as np
import easyocr
import re
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import os
import pickle
from PIL import Image
import pytesseract
from fuzzywuzzy import fuzz, process
import cohere
import uuid
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix PIL.Image.ANTIALIAS deprecation issue
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

@dataclass
class Patient:
    name: str = ""
    age: str = ""
    gender: str = ""

@dataclass
class Doctor:
    name: str = ""
    specialization: str = ""
    registration_number: str = ""

@dataclass
class Medicine:
    name: str = ""
    dosage: str = ""
    quantity: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    available: bool = True

@dataclass
class AnalysisResult:
    prescription_id: str
    patient: Patient
    doctor: Doctor
    medicines: List[Medicine]
    diagnosis: List[str]
    confidence_score: float
    raw_text: str
    success: bool = True
    error: str = ""

class EnhancedPrescriptionAnalyzer:
    def __init__(self, cohere_api_key: str = None, tesseract_path: str = None, force_api: bool = True):
        """Initialize the analyzer"""
        # Initialize Cohere API
        self._init_cohere_api(cohere_api_key, force_api)
        
        # Initialize OCR readers
        self._init_ocr_readers()
        
        # Load medicine database
        self.medicine_database = self._load_medicine_database()
        if not self.medicine_database:
            self.medicine_database = self._create_default_medicine_database()
            self._save_medicine_database()

        # Enhanced medical patterns
        self.doctor_patterns = {
            'titles': [
                r'\b(Dr\.?|Doctor|Prof\.?|Professor)\s+([A-Za-z\s\.]+)',
                r'\b(MBBS|MD|MS|DM|MCh|FRCS|MRCP|DNB|Dip\.?)\b',
                r'\b(Consultant|Senior\s+Consultant|Associate\s+Professor|Professor)\b'
            ],
            'specializations': [
                r'\b(Cardiologist|Neurologist|Orthopedic|Pediatrician|Dermatologist|Gynecologist|Psychiatrist|Radiologist|Anesthesiologist|Pathologist|Oncologist|Urologist|ENT|Ophthalmologist|General\s+Medicine|Internal\s+Medicine|Emergency\s+Medicine)\b',
                r'\b(Cardiology|Neurology|Orthopedics|Pediatrics|Dermatology|Gynecology|Psychiatry|Radiology|Anesthesiology|Pathology|Oncology|Urology|Ophthalmology)\b'
            ],
            'registration': [
                r'\b(Reg\.?\s*No\.?|Registration\s+No\.?|License\s+No\.?|Medical\s+License)\s*:?\s*([A-Z0-9]+)',
                r'\b([A-Z]{2,4}[-/]?\d{4,6})\b'
            ]
        }
        
        self.patient_patterns = {
            'age_indicators': [
                r'\b(Age|age)\s*:?\s*(\d{1,3})\s*(years?|yrs?|Y)?',
                r'\b(\d{1,3})\s*(years?|yrs?|Y)\s*(old|age)?',
                r'\b(Age|age)\s*[-:]\s*(\d{1,3})'
            ],
            'gender_indicators': [
                r'\b(Male|Female|M|F|male|female)\b',
                r'\b(Gender|Sex)\s*:?\s*(Male|Female|M|F)',
                r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+([A-Za-z\s]+)'
            ],
            'name_patterns': [
                r'\b(Patient|patient)\s*:?\s*([A-Za-z\s\.]+)',
                r'\b(Name|name)\s*:?\s*([A-Za-z\s\.]+)',
                r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+([A-Za-z\s]+)'
            ]
        }

        self.medical_abbreviations = {
            'bd': 'twice daily', 'bid': 'twice daily', 'tid': 'three times daily',
            'qid': 'four times daily', 'od': 'once daily', 'qd': 'once daily',
            'sos': 'as needed', 'prn': 'as needed', 'ac': 'before meals',
            'pc': 'after meals', 'hs': 'at bedtime', 'qhs': 'at bedtime',
            'mg': 'milligrams', 'gm': 'grams', 'g': 'grams', 'ml': 'milliliters',
            'cap': 'capsule', 'tab': 'tablet', 'syp': 'syrup', 'inj': 'injection'
        }

    def _init_cohere_api(self, cohere_api_key: str, force_api: bool):
        """Initialize Cohere API with proper error handling"""
        key_from_file = None
        try:
            from integration.keys import COHERE_API_KEY as KEY_FILE
            key_from_file = KEY_FILE
            logger.info("✓ Cohere API key loaded from integration.keys")
        except ImportError:
            logger.warning("⚠ Could not import keys.py (integration/keys.py)")

        api_key = cohere_api_key or os.getenv("COHERE_API_KEY") or key_from_file

        self.co = None
        if api_key:
            try:
                self.co = cohere.Client(api_key)
                logger.info("✓ Cohere API client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Cohere client: {e}")
                if force_api:
                    raise ValueError(f"Failed to initialize Cohere API: {e}")
        else:
            error_msg = "No Cohere API key provided"
            logger.error(f"❌ {error_msg}")
            if force_api:
                raise ValueError(error_msg)

    def _init_ocr_readers(self):
        """Initialize OCR readers with error handling"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✓ EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_reader = None

    def _load_medicine_database(self):
        """Load medicine database from file"""
        try:
            with open("medicine_database.pkl", "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return None

    def _save_medicine_database(self):
        """Save medicine database to file"""
        try:
            with open("medicine_database.pkl", "wb") as f:
                pickle.dump(self.medicine_database, f)
            logger.info("Medicine database saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save medicine database: {e}")

    def _create_default_medicine_database(self):
        """Return a comprehensive medicine database dict"""
        return {
            # Antibiotics
            'augmentin': {'category': 'antibiotic', 'generic': 'amoxicillin + clavulanic acid', 'available': True},
            'amoxicillin': {'category': 'antibiotic', 'generic': 'amoxicillin', 'available': True},
            'azithromycin': {'category': 'antibiotic', 'generic': 'azithromycin', 'available': True},
            'ciprofloxacin': {'category': 'antibiotic', 'generic': 'ciprofloxacin', 'available': True},
            'cephalexin': {'category': 'antibiotic', 'generic': 'cephalexin', 'available': True},
            'doxycycline': {'category': 'antibiotic', 'generic': 'doxycycline', 'available': True},
            
            # Pain relievers
            'paracetamol': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'acetaminophen': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'ibuprofen': {'category': 'nsaid', 'generic': 'ibuprofen', 'available': True},
            'diclofenac': {'category': 'nsaid', 'generic': 'diclofenac', 'available': True},
            'aspirin': {'category': 'nsaid', 'generic': 'aspirin', 'available': True},
            
            # PPIs and antacids
            'esomeprazole': {'category': 'ppi', 'generic': 'esomeprazole', 'available': True},
            'omeprazole': {'category': 'ppi', 'generic': 'omeprazole', 'available': True},
            'pantoprazole': {'category': 'ppi', 'generic': 'pantoprazole', 'available': True},
            
            # Antihistamines
            'cetirizine': {'category': 'antihistamine', 'generic': 'cetirizine', 'available': True},
            'loratadine': {'category': 'antihistamine', 'generic': 'loratadine', 'available': True},
            
            # Common Indian medicines
            'crocin': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'combiflam': {'category': 'analgesic', 'generic': 'ibuprofen + paracetamol', 'available': True},
            'dolo': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
        }

    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """
        FIXED: Enhanced image preprocessing with proper grayscale conversion
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return []

        try:
            # CRITICAL FIX: Convert to grayscale FIRST
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Resize if too small
            height, width = gray.shape
            if height < 800 or width < 600:
                scale_factor = max(800/height, 600/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            processed_images = []

            # Method 1: CLAHE + Adaptive Threshold
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                contrast_enhanced = clahe.apply(gray)  # gray is already grayscale
                denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
                adaptive_thresh = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(adaptive_thresh)
                logger.info("✓ Method 1 (CLAHE) preprocessing successful")
            except Exception as e:
                logger.warning(f"Method 1 preprocessing failed: {e}")

            # Method 2: Otsu's Thresholding
            try:
                blur = cv2.GaussianBlur(gray, (3,3), 0)  # gray is already grayscale
                _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(otsu_thresh)
                logger.info("✓ Method 2 (Otsu) preprocessing successful")
            except Exception as e:
                logger.warning(f"Method 2 preprocessing failed: {e}")

            # Method 3: Simple binary threshold (fallback)
            try:
                _, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                processed_images.append(simple_thresh)
                logger.info("✓ Method 3 (Simple threshold) preprocessing successful")
            except Exception as e:
                logger.warning(f"Method 3 preprocessing failed: {e}")

            if not processed_images:
                logger.warning("All preprocessing methods failed, using original grayscale")
                return [gray]
            
            return processed_images

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return []

    def extract_text(self, processed_images: List[np.ndarray]) -> Tuple[str, float]:
        """
        FIXED: Extract text using multiple OCR methods
        Returns: (text, confidence) - removed third return value for consistency
        """
        all_results = []
        all_confidences = []

        for i, image in enumerate(processed_images):
            # EasyOCR
            if self.easyocr_reader:
                try:
                    easyocr_results = self.easyocr_reader.readtext(image, detail=1)
                    if easyocr_results:
                        text = " ".join([r[1] for r in easyocr_results])
                        confidence = np.mean([r[2] for r in easyocr_results])
                        all_results.append(("EasyOCR", text, confidence))
                        all_confidences.append(confidence)
                        logger.info(f"✓ EasyOCR extraction successful (confidence: {confidence:.2f})")
                except Exception as e:
                    logger.warning(f"EasyOCR failed for image {i+1}: {e}")

            # Tesseract with multiple configurations
            tesseract_configs = [
                '--oem 3 --psm 6',  # Uniform text block
                '--oem 3 --psm 3',  # Fully automatic
                '--oem 3 --psm 4',  # Single column
            ]

            for j, config in enumerate(tesseract_configs):
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                    conf_scores = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if conf_scores and len(text.strip()) > 10:
                        confidence = np.mean(conf_scores) / 100
                        all_results.append((f"Tesseract_c{j+1}", text, confidence))
                        all_confidences.append(confidence)
                        logger.info(f"✓ Tesseract config {j+1} successful (confidence: {confidence:.2f})")
                        if confidence > 0.8:
                            break
                except Exception as e:
                    logger.warning(f"Tesseract config {j+1} failed: {e}")

        if not all_results:
            logger.error("All OCR methods failed")
            return "", 0.0

        # Select best results
        sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
        
        # Combine top results
        combined_texts = []
        seen_lines = set()
        
        for _, text, conf in sorted_results[:4]:
            if conf > 0.3:
                cleaned = self._clean_text(text)
                if cleaned:
                    lines = cleaned.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line not in seen_lines and len(line) > 2:
                            combined_texts.append(line)
                            seen_lines.add(line)

        if not combined_texts:
            return "", 0.0

        final_text = "\n".join(combined_texts)
        overall_confidence = np.mean(all_confidences) if all_confidences else 0.0

        logger.info(f"✓ Final text extraction: {len(final_text)} chars, confidence: {overall_confidence:.2f}")
        
        return final_text, overall_confidence

    def _clean_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\:\(\)\-\/\+\&\'\"]', '', text)

        # Fix common OCR mistakes
        fixes = {
            r'\b0\b': 'O', r'\b1\b': 'I', r'rng': 'mg', r'\.mg': ' mg',
            r'Tab\b': 'Tab', r'Cap\b': 'Cap', r'\bBd\b': 'bd', r'\bOd\b': 'od',
            r'\bSyp\b': 'Syp', r'\bInj\b': 'Inj', r'rnl': 'ml', r'gm\b': 'gm',
            r'\b5rng\b': '5mg', r'\b10rng\b': '10mg', r'\b25rng\b': '25mg',
            r'Dr\s*\.?': 'Dr.', r'Mrs?\s*\.?': 'Mr.', r'Mis+\s*\.?': 'Miss',
            r'(\d+)\s*x\s*(\d+)': r'\1 x \2',
            r'(\d+)\s*mg': r'\1 mg',
            r'(\d+)\s*ml': r'\1 ml',
        }

        for pattern, repl in fixes.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text.strip()

    def extract_doctor_patient_info(self, text: str) -> Tuple[Dict, Dict]:
        """Extract doctor and patient information using pattern matching"""
        doctor_info = {'name': '', 'specialization': '', 'registration_number': ''}
        patient_info = {'name': '', 'age': '', 'gender': ''}
        
        # Extract doctor information
        for pattern in self.doctor_patterns['titles']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2 and not doctor_info['name']:
                    doctor_info['name'] = match.group(2).strip()
        
        for pattern in self.doctor_patterns['specializations']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['specialization']:
                doctor_info['specialization'] = match.group(0).strip()
        
        for pattern in self.doctor_patterns['registration']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['registration_number']:
                if len(match.groups()) >= 2:
                    doctor_info['registration_number'] = match.group(2).strip()
                else:
                    doctor_info['registration_number'] = match.group(1).strip()
        
        # Extract patient information
        for pattern in self.patient_patterns['age_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['age']:
                for group in match.groups():
                    if group and group.isdigit():
                        patient_info['age'] = group
                        break
        
        for pattern in self.patient_patterns['gender_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['gender']:
                gender_text = match.group(0).lower()
                if 'male' in gender_text or 'm' in gender_text:
                    patient_info['gender'] = 'Male' if 'female' not in gender_text else 'Female'
                elif 'female' in gender_text or 'f' in gender_text:
                    patient_info['gender'] = 'Female'
        
        for pattern in self.patient_patterns['name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['name']:
                if len(match.groups()) >= 2:
                    name_candidate = match.group(2).strip()
                    if 2 <= len(name_candidate) <= 50 and re.search(r'[A-Za-z]', name_candidate):
                        patient_info['name'] = name_candidate
        
        if not patient_info['name']:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 2 and len(line) < 50 and re.match(r'^[A-Za-z\s\.]+$', line):
                    if not any(term in line.lower() for term in ['dr.', 'doctor', 'clinic', 'hospital', 'prescription', 'medicine']):
                        if not patient_info['name']:
                            patient_info['name'] = line
        
        return doctor_info, patient_info

    def _extract_medicines_simple(self, text: str) -> List[Medicine]:
        """Extract medicines using simple pattern matching"""
        medicines = []
        lines = text.split('\n')
        
        medicine_keywords = ['tab', 'cap', 'syp', 'inj', 'tablet', 'capsule', 'syrup', 'injection']
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['patient', 'doctor', 'date', 'age', 'gender', 'diagnosis']):
                continue
            
            medicine_match = re.search(
                r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(\d+\s*(?:mg|ml|gm|g|mcg))',
                line,
                re.IGNORECASE
            )
            
            if medicine_match:
                name = medicine_match.group(1).strip()
                dosage = medicine_match.group(2).strip()
                
                if len(name) < 3 or name.lower() in ['tab', 'cap', 'syp', 'the', 'and', 'for']:
                    continue
                
                frequency = self._extract_frequency(line)
                duration = self._extract_duration(line)
                instructions = self._extract_instructions(line)
                
                medicine = Medicine(
                    name=name,
                    dosage=dosage,
                    quantity="1",
                    frequency=frequency,
                    duration=duration,
                    instructions=instructions,
                    available=self._check_availability(name)
                )
                medicines.append(medicine)
                continue
            
            tab_match = re.search(
                r'(Tab\.?|Cap\.?|Syp\.?|Inj\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
                line,
                re.IGNORECASE
            )
            
            if tab_match:
                name = tab_match.group(2).strip()
                
                dosage_match = re.search(r'(\d+\s*(?:mg|ml|gm|g|mcg))', line, re.IGNORECASE)
                dosage = dosage_match.group(1) if dosage_match else "As prescribed"
                
                frequency = self._extract_frequency(line)
                duration = self._extract_duration(line)
                instructions = self._extract_instructions(line)
                
                medicine = Medicine(
                    name=name,
                    dosage=dosage,
                    quantity="1",
                    frequency=frequency,
                    duration=duration,
                    instructions=instructions,
                    available=self._check_availability(name)
                )
                medicines.append(medicine)
        
        # Remove duplicates
        unique_medicines = []
        seen_names = set()
        
        for med in medicines:
            name_lower = med.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_medicines.append(med)
        
        return unique_medicines

    def _extract_frequency(self, text: str) -> str:
        """Extract medication frequency"""
        text_lower = text.lower()
        
        frequency_patterns = {
            r'\b(once\s+daily|od|qd|1\s*-\s*0\s*-\s*0)\b': 'Once daily',
            r'\b(twice\s+daily|bd|bid|2\s*times|1\s*-\s*0\s*-\s*1)\b': 'Twice daily',
            r'\b(thrice\s+daily|tid|3\s*times|1\s*-\s*1\s*-\s*1)\b': 'Three times daily',
            r'\b(four\s+times|qid|1\s*-\s*1\s*-\s*1\s*-\s*1)\b': 'Four times daily',
            r'\b(as\s+needed|sos|prn)\b': 'As needed',
            r'\b(at\s+bedtime|hs|qhs|0\s*-\s*0\s*-\s*1)\b': 'At bedtime',
        }
        
        for pattern, frequency in frequency_patterns.items():
            if re.search(pattern, text_lower):
                return frequency
        
        return "As directed"

    def _extract_duration(self, text: str) -> str:
        """Extract medication duration"""
        duration_patterns = [
            r'(\d+\s*(?:day|days|d))',
            r'(\d+\s*(?:week|weeks|wk|wks))',
            r'(\d+\s*(?:month|months|mo|mos))',
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "As prescribed"

    def _extract_instructions(self, text: str) -> str:
        """Extract medication instructions"""
        text_lower = text.lower()
        instructions = []
        
        if re.search(r'\b(before\s+(?:food|meal|eating))\b', text_lower):
            instructions.append("Before meals")
        elif re.search(r'\b(after\s+(?:food|meal|eating))\b', text_lower):
            instructions.append("After meals")
        elif re.search(r'\b(with\s+(?:food|meal))\b', text_lower):
            instructions.append("With meals")
        
        return ", ".join(instructions) if instructions else ""

    def _check_availability(self, medicine_name: str) -> bool:
        """Check medicine availability"""
        if not medicine_name:
            return True
            
        medicine_lower = medicine_name.lower().strip()
        
        if medicine_lower in self.medicine_database:
            return self.medicine_database[medicine_lower]['available']
        
        best_match = process.extractOne(medicine_lower, self.medicine_database.keys(), scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 75:
            return self.medicine_database[best_match[0]]['available']
        
        return True

    def _calculate_confidence(self, ocr_confidence: float, medicines_count: int, 
                            patient: Patient, doctor: Doctor) -> float:
        """Calculate overall confidence score"""
        weights = {'ocr': 0.3, 'medicines': 0.3, 'patient_info': 0.2, 'doctor_info': 0.2}
        
        medicine_score = 0.0
        if medicines_count > 0:
            medicine_score = min(1.0, medicines_count / 3.0)
            medicine_score = min(1.0, medicine_score + 0.3)
        
        patient_score = 0.0
        if patient.name: patient_score += 0.5
        if patient.age: patient_score += 0.25
        if patient.gender: patient_score += 0.25
        patient_score = min(1.0, patient_score)
        
        doctor_score = 0.0
        if doctor.name: doctor_score += 0.6
        if doctor.specialization: doctor_score += 0.2
        if doctor.registration_number: doctor_score += 0.2
        doctor_score = min(1.0, doctor_score)
        
        total_score = (
            weights['ocr'] * ocr_confidence +
            weights['medicines'] * medicine_score +
            weights['patient_info'] * patient_score +
            weights['doctor_info'] * doctor_score
        )
        
        return min(1.0, max(0.0, total_score))

    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """
        FIXED: Main method to analyze prescription image
        """
        try:
            prescription_id = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}{str(uuid.uuid4())[:8]}"
            
            logger.info(f"Starting analysis for prescription {prescription_id}")
            
            # Preprocess image
            processed_images = self.preprocess_image(image_path)
            if not processed_images:
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(), medicines=[],
                    diagnosis=[], confidence_score=0.0, raw_text="",
                    success=False, error="Failed to preprocess image"
                )
            
            # Extract text - FIXED: now returns only 2 values
            extracted_text, ocr_confidence = self.extract_text(processed_images)
            if not extracted_text.strip():
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(), medicines=[],
                    diagnosis=[], confidence_score=0.0, raw_text="",
                    success=False, error="No text could be extracted"
                )
            
            logger.info(f"✓ Extracted {len(extracted_text)} characters with {ocr_confidence:.2f} confidence")
            
            # Clean text
            cleaned_text = self._clean_text(extracted_text)
            
            # Extract doctor and patient info
            doctor_info, patient_info = self.extract_doctor_patient_info(cleaned_text)
            
            # Create patient object
            patient = Patient(
                name=patient_info.get('name', ''),
                age=patient_info.get('age', ''),
                gender=patient_info.get('gender', '')
            )
            
            # Create doctor object
            doctor = Doctor(
                name=doctor_info.get('name', ''),
                specialization=doctor_info.get('specialization', ''),
                registration_number=doctor_info.get('registration_number', '')
            )
            
            # Extract medicines
            medicines = self._extract_medicines_simple(cleaned_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(ocr_confidence, len(medicines), patient, doctor)
            
            logger.info(f"✓ Analysis complete: {confidence:.2%} confidence")
            logger.info(f"  Doctor: {doctor.name}")
            logger.info(f"  Patient: {patient.name}")
            logger.info(f"  Medicines: {len(medicines)}")
            
            # Create successful result
            return AnalysisResult(
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                diagnosis=[],
                confidence_score=confidence,
                raw_text=cleaned_text,
                success=True,
                error=""
            )
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_prescription: {e}")
            import traceback
            traceback.print_exc()
            
            return AnalysisResult(
                prescription_id=f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}",
                patient=Patient(), 
                doctor=Doctor(),
                medicines=[], 
                diagnosis=[],
                confidence_score=0.0, 
                raw_text="",
                success=False, 
                error=str(e)
            )

    def to_json(self, result: AnalysisResult) -> Dict:
        """Convert AnalysisResult to JSON format expected by FastAPI"""
        return {
            "success": result.success,
            "prescription_id": result.prescription_id,
            "patient": {
                "name": result.patient.name,
                "age": result.patient.age,
                "gender": result.patient.gender
            },
            "doctor": {
                "name": result.doctor.name,
                "specialization": result.doctor.specialization,
                "registration_number": result.doctor.registration_number
            },
            "medicines": [
                {
                    "name": med.name,
                    "dosage": med.dosage,
                    "quantity": med.quantity,
                    "frequency": med.frequency,
                    "duration": med.duration,
                    "instructions": med.instructions,
                    "available": med.available
                } for med in result.medicines
            ],
            "diagnosis": result.diagnosis,
            "confidence_score": result.confidence_score,
            "message": "Analysis completed successfully" if result.success else result.error,
            "error": result.error if not result.success else "",
            "raw_text": result.raw_text
        }