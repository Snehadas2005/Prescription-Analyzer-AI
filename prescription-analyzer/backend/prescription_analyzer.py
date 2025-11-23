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

@dataclass
class MedicationItem(BaseModel):
    name: str = Field(default="", description="Medication name")
    dosage: str = Field(default="", description="Medication dosage")
    frequency: str = Field(default="", description="Frequency of medication")
    duration: str = Field(default="", description="Duration of medication")

class EnhancedPrescriptionAnalyzer:
    def __init__(self, cohere_api_key: str = None, tesseract_path: str = None, force_api: bool = True):
        """
        Initialize the analyzer with enhanced doctor/patient detection
        """
        # Initialize Cohere API
        self._init_cohere_api(cohere_api_key, force_api)
        
        # Initialize OCR readers
        self._init_ocr_readers()
        
        # Load medicine database
        self.medicine_database = self._load_medicine_database()
        if not self.medicine_database:
            self.medicine_database = self._create_default_medicine_database()
            self._save_medicine_database()

        # Enhanced medical patterns for better doctor/patient identification
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
                r'\b([A-Z]{2,4}[-/]?\d{4,6})\b'  # Common registration number patterns
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

        # Medical abbreviations mapping
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
            logger.info("Cohere API key loaded from integration.keys ✅")
        except ImportError:
            logger.warning("⚠ Could not import keys.py (integration/keys.py)")

        api_key = cohere_api_key or os.getenv("COHERE_API_KEY") or key_from_file

        self.co = None
        if api_key:
            try:
                self.co = cohere.Client(api_key)
                logger.info("Cohere API client initialized successfully 🎉")
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
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_reader = None

    def _load_medicine_database(self):
        """Load medicine database from file or create new one"""
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
            'clarithromycin': {'category': 'antibiotic', 'generic': 'clarithromycin', 'available': True},
            
            # Pain relievers
            'paracetamol': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'acetaminophen': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'ibuprofen': {'category': 'nsaid', 'generic': 'ibuprofen', 'available': True},
            'diclofenac': {'category': 'nsaid', 'generic': 'diclofenac', 'available': True},
            'aspirin': {'category': 'nsaid', 'generic': 'aspirin', 'available': True},
            'naproxen': {'category': 'nsaid', 'generic': 'naproxen', 'available': True},
            
            # PPIs and antacids
            'esomeprazole': {'category': 'ppi', 'generic': 'esomeprazole', 'available': True},
            'omeprazole': {'category': 'ppi', 'generic': 'omeprazole', 'available': True},
            'pantoprazole': {'category': 'ppi', 'generic': 'pantoprazole', 'available': True},
            'lansoprazole': {'category': 'ppi', 'generic': 'lansoprazole', 'available': True},
            'ranitidine': {'category': 'h2_blocker', 'generic': 'ranitidine', 'available': False},
            
            # Antihistamines
            'cetirizine': {'category': 'antihistamine', 'generic': 'cetirizine', 'available': True},
            'loratadine': {'category': 'antihistamine', 'generic': 'loratadine', 'available': True},
            'fexofenadine': {'category': 'antihistamine', 'generic': 'fexofenadine', 'available': True},
            
            # Common Indian medicines from prescriptions
            'gan vi': {'category': 'ayurvedic', 'generic': 'herbal supplement', 'available': True},
            'crocin': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'combiflam': {'category': 'analgesic', 'generic': 'ibuprofen + paracetamol', 'available': True},
            'dolo': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'volini': {'category': 'topical', 'generic': 'topical analgesic', 'available': True},
            
            # Diabetes medications
            'metformin': {'category': 'antidiabetic', 'generic': 'metformin', 'available': True},
            'insulin': {'category': 'antidiabetic', 'generic': 'insulin', 'available': True},
            'glimepiride': {'category': 'antidiabetic', 'generic': 'glimepiride', 'available': True},
            
            # Vitamins and supplements
            'vitamin d3': {'category': 'vitamin', 'generic': 'cholecalciferol', 'available': True},
            'vitamin b12': {'category': 'vitamin', 'generic': 'cyanocobalamin', 'available': True},
            'iron': {'category': 'mineral', 'generic': 'ferrous sulfate', 'available': True},
            'calcium': {'category': 'mineral', 'generic': 'calcium carbonate', 'available': True}
        }

    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """Enhanced image preprocessing for better OCR results"""
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
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
                contrast_enhanced = clahe.apply(gray)
                denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
                adaptive_thresh = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(adaptive_thresh)
            except Exception as e:
                logger.warning(f"Method 1 preprocessing failed: {e}")

            # Method 2: Otsu's Thresholding
            try:
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(otsu_thresh)
            except Exception as e:
                logger.warning(f"Method 2 preprocessing failed: {e}")

            # Method 3: Edge-preserving filter + threshold
            try:
                filtered = cv2.edgePreservingFilter(gray, flags=2, sigma_s=50, sigma_r=0.4)
                _, edge_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(edge_thresh)
            except Exception as e:
                logger.warning(f"Method 3 preprocessing failed: {e}")

            return processed_images if processed_images else [gray]

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return []

    def extract_text(self, processed_images: List[np.ndarray]) -> Tuple[str, float]:
        """Extract text using multiple OCR methods with enhanced results"""
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
                except Exception as e:
                    logger.warning(f"EasyOCR failed for image {i+1}: {e}")

            # Tesseract with multiple configurations
            tesseract_configs = [
                '--oem 3 --psm 6',  # Uniform text block
                '--oem 3 --psm 3',  # Fully automatic page segmentation
                '--oem 3 --psm 4',  # Single column text
                '--oem 3 --psm 11', # Sparse text
                '--oem 3 --psm 12'  # Single text line
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
                        if confidence > 0.8:
                            break
                except Exception as e:
                    logger.warning(f"Tesseract config {j+1} failed: {e}")

        if not all_results:
            return "", 0.0

        # Select best results and combine them intelligently
        sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
        
        # Combine top results with deduplication
        combined_texts = []
        seen_lines = set()
        
        for _, text, conf in sorted_results[:4]:  # Use top 4 results
            if conf > 0.3:
                cleaned = self._clean_text(text)
                if cleaned:
                    # Split into lines and deduplicate
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

        return final_text, overall_confidence

    def _clean_text(self, text: str) -> str:
        """Clean and normalize OCR text with enhanced corrections"""
        if not text:
            return ""

        # Remove extra whitespace and unwanted characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\:\(\)\-\/\+\&\'\"]', '', text)

        # Fix common OCR mistakes for medical prescriptions
        fixes = {
            r'\b0\b': 'O', r'\b1\b': 'I', r'rng': 'mg', r'\.mg': ' mg',
            r'Tab\b': 'Tab', r'Cap\b': 'Cap', r'\bBd\b': 'bd', r'\bOd\b': 'od',
            r'\bSyp\b': 'Syp', r'\bInj\b': 'Inj', r'rnl': 'ml', r'gm\b': 'gm',
            r'\b5rng\b': '5mg', r'\b10rng\b': '10mg', r'\b25rng\b': '25mg',
            r'Dr\s*\.?': 'Dr.', r'Mrs?\s*\.?': 'Mr.', r'Mis+\s*\.?': 'Miss',
            # Common prescription format fixes
            r'(\d+)\s*x\s*(\d+)': r'\1 x \2',  # Fix dosage format
            r'(\d+)\s*mg': r'\1 mg',  # Ensure space before mg
            r'(\d+)\s*ml': r'\1 ml',  # Ensure space before ml
        }

        for pattern, repl in fixes.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text.strip()

    def extract_doctor_patient_info(self, text: str) -> Tuple[Dict, Dict]:
        """Enhanced extraction of doctor and patient information using pattern matching"""
        doctor_info = {'name': '', 'specialization': '', 'registration_number': ''}
        patient_info = {'name': '', 'age': '', 'gender': ''}
        
        # Extract doctor information
        for pattern in self.doctor_patterns['titles']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2 and not doctor_info['name']:
                    doctor_info['name'] = match.group(2).strip()
        
        # Extract specializations
        for pattern in self.doctor_patterns['specializations']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['specialization']:
                doctor_info['specialization'] = match.group(0).strip()
        
        # Extract registration numbers
        for pattern in self.doctor_patterns['registration']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['registration_number']:
                if len(match.groups()) >= 2:
                    doctor_info['registration_number'] = match.group(2).strip()
                else:
                    doctor_info['registration_number'] = match.group(1).strip()
        
        # Extract patient information
        # Patient age
        for pattern in self.patient_patterns['age_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['age']:
                # Find the group that contains the age number
                for group in match.groups():
                    if group and group.isdigit():
                        patient_info['age'] = group
                        break
        
        # Patient gender
        for pattern in self.patient_patterns['gender_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['gender']:
                gender_text = match.group(0).lower()
                if 'male' in gender_text or 'm' in gender_text:
                    patient_info['gender'] = 'Male' if 'female' not in gender_text else 'Female'
                elif 'female' in gender_text or 'f' in gender_text:
                    patient_info['gender'] = 'Female'
        
        # Patient name - more sophisticated extraction
        for pattern in self.patient_patterns['name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['name']:
                if len(match.groups()) >= 2:
                    name_candidate = match.group(2).strip()
                    # Validate name (should be reasonable length and contain letters)
                    if 2 <= len(name_candidate) <= 50 and re.search(r'[A-Za-z]', name_candidate):
                        patient_info['name'] = name_candidate
        
        # If no explicit patient name found, try to infer from lines that might contain names
        if not patient_info['name']:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines that might contain patient names
                if len(line) > 2 and len(line) < 50 and re.match(r'^[A-Za-z\s\.]+$', line):
                    # Skip if it looks like a doctor's name or medical term
                    if not any(term in line.lower() for term in ['dr.', 'doctor', 'clinic', 'hospital', 'prescription', 'medicine']):
                        if not patient_info['name']:  # Take first reasonable candidate
                            patient_info['name'] = line
        
        return doctor_info, patient_info

    def analyze_with_cohere(self, extracted_text: str, ocr_confidence: float) -> Dict:
        """Analyze prescription using Cohere API with enhanced doctor/patient detection"""
        if not self.co:
            raise ValueError("Cohere API client not initialized. Please provide a valid API key.")
        
        return self._analyze_with_cohere_api(extracted_text, ocr_confidence)

    def _analyze_with_cohere_api(self, extracted_text: str, ocr_confidence: float) -> Dict:
        """Enhanced Cohere API analysis with better prompting for doctor/patient identification"""
        
        # First, try pattern-based extraction
        doctor_info, patient_info = self.extract_doctor_patient_info(extracted_text)
        
        prompt = f"""
        You are an expert medical transcriptionist. Extract prescription information from this text and return ONLY valid JSON.

        CRITICAL INSTRUCTIONS:
        1. Carefully distinguish between DOCTOR and PATIENT information
        2. Doctor names often have titles (Dr., Prof.) or qualifications (MBBS, MD, etc.)
        3. Patient names are usually simpler, without medical titles
        4. Look for context clues like "Patient:", "Name:", age indicators, gender markers
        
        Pre-identified information to verify/correct:
        Doctor: {doctor_info}
        Patient: {patient_info}

        Return ONLY this JSON structure:
        {{
            "patient_name": "",
            "patient_age": null,
            "patient_gender": "",
            "doctor_name": "",
            "doctor_license": "",
            "doctor_specialization": "",
            "prescription_date": "YYYY-MM-DD",
            "medications": [
                {{
                    "name": "",
                    "dosage": "",
                    "frequency": "",
                    "duration": ""
                }}
            ],
            "diagnosis": "",
            "additional_notes": ""
        }}

        Rules:
        - Use null for patient_age if not found or unclear
        - Use empty strings for missing text fields
        - Extract ALL medications found
        - Be precise with dosages, frequencies, and durations
        - Include both generic and brand names when available
        - Distinguish clearly between doctor and patient names

        Prescription text:
        {extracted_text}
        """

        try:
            response = self.co.chat(
                model="command-r",
                message="Return ONLY valid JSON. Do not add json blocks, backticks, or any explanation.\n\n" + prompt
            )

            # Handle different Cohere SDK response formats
            raw_text = ""
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'message'):
                raw_text = response.message
            elif hasattr(response, 'content'):
                raw_text = response.content
            else:
                if isinstance(response, dict):
                    raw_text = response.get('text', '') or response.get('message', '') or response.get('content', '')
                else:
                    raw_text = str(response)

            if not raw_text:
                raise ValueError("Empty response from Cohere API")

            raw_text = raw_text.strip()

            # Clean JSON fences if present
            if raw_text.startswith("```"):
                lines = raw_text.split('\n')
                if lines[0].startswith("```") and lines[-1].strip() == "```":
                    raw_text = '\n'.join(lines[1:-1])
                elif lines[0].startswith("```json"):
                    raw_text = '\n'.join(lines[1:])
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]

            # Parse JSON
            data = json.loads(raw_text)

            # Validate and enhance the extracted data
            data = self._validate_and_enhance_extraction(data, doctor_info, patient_info)

            # Add metadata
            data['ocr_confidence'] = ocr_confidence
            data['raw_text'] = extracted_text
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Raw response: {raw_text}")
            # Fallback to pattern-based extraction
            return self._fallback_extraction(extracted_text, ocr_confidence, doctor_info, patient_info)
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            # Fallback to pattern-based extraction
            return self._fallback_extraction(extracted_text, ocr_confidence, doctor_info, patient_info)

    def _validate_and_enhance_extraction(self, data: Dict, doctor_info: Dict, patient_info: Dict) -> Dict:
        """Validate and enhance the extracted data using pattern-based results"""
        
        # Use pattern-based results as fallback if API results are empty/poor
        if not data.get('doctor_name') and doctor_info.get('name'):
            data['doctor_name'] = doctor_info['name']
        if not data.get('doctor_license') and doctor_info.get('registration_number'):
            data['doctor_license'] = doctor_info['registration_number']
        if not data.get('doctor_specialization') and doctor_info.get('specialization'):
            data['doctor_specialization'] = doctor_info['specialization']
            
        if not data.get('patient_name') and patient_info.get('name'):
            data['patient_name'] = patient_info['name']
        if not data.get('patient_age') and patient_info.get('age'):
            try:
                data['patient_age'] = int(patient_info['age'])
            except (ValueError, TypeError):
                data['patient_age'] = None
        if not data.get('patient_gender') and patient_info.get('gender'):
            data['patient_gender'] = patient_info['gender']
        
        # Sanitize the data
        data = self.sanitize_prescription_data(data)
        
        return data

    def _fallback_extraction(self, extracted_text: str, ocr_confidence: float, doctor_info: Dict, patient_info: Dict) -> Dict:
        """Fallback extraction using pattern matching when API fails"""
        logger.info("Using pattern-based fallback extraction")
        
        # Extract medicines using simple patterns
        medicines = self._extract_medicines_pattern_based(extracted_text)
        
        data = {
            'patient_name': patient_info.get('name', ''),
            'patient_age': int(patient_info['age']) if patient_info.get('age') and patient_info['age'].isdigit() else None,
            'patient_gender': patient_info.get('gender', ''),
            'doctor_name': doctor_info.get('name', ''),
            'doctor_license': doctor_info.get('registration_number', ''),
            'doctor_specialization': doctor_info.get('specialization', ''),
            'prescription_date': datetime.now().strftime('%Y-%m-%d'),
            'medications': medicines,
            'diagnosis': '',
            'additional_notes': '',
            'ocr_confidence': ocr_confidence,
            'raw_text': extracted_text
        }
        
        return self.sanitize_prescription_data(data)

    def _extract_medicines_pattern_based(self, text: str) -> List[Dict]:
        """Extract medicines using pattern matching"""
        medicines = []
        lines = text.split('\n')
        
        # Common medicine patterns
        medicine_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(\d+\s*(?:mg|ml|gm|g))\s+(\w+)\s*(?:x\s*(\d+))?',  # Name Dosage Frequency x Duration
            r'(\w+(?:\s+\w+)*)\s+(\d+)\s*(?:mg|ml|gm|g)\s+(\w+)',  # Name Dosage Frequency
            r'(\w+(?:\s+\w+)*)\s+(?:Tab|Cap|Syp)\s+(\d+)\s*(?:mg|ml)',  # Name Tab/Cap/Syp Dosage
            r'(\w+)\s+(\d+)\s*(od|bd|tid|qid|sos)',  # Simple Name Dosage Frequency
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
                
            for pattern in medicine_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    medicine = {
                        'name': groups[0].strip() if groups[0] else '',
                        'dosage': f"{groups[1]} {groups[2] if len(groups) > 2 else ''}" if len(groups) > 1 else '',
                        'frequency': groups[2] if len(groups) > 2 else '',
                        'duration': groups[3] if len(groups) > 3 and groups[3] else ''
                    }
                    
                    # Clean up the medicine entry
                    if medicine['name'] and len(medicine['name']) > 1:
                        # Expand abbreviations
                        if medicine['frequency'].lower() in self.medical_abbreviations:
                            medicine['frequency'] = self.medical_abbreviations[medicine['frequency'].lower()]
                        
                        medicines.append(medicine)
                    break
        
        # If no medicines found with patterns, try simple word extraction
        if not medicines:
            for line in lines:
                line = line.strip()
                # Look for lines that might contain medicine names
                if (len(line) > 2 and len(line) < 50 and 
                    not any(keyword in line.lower() for keyword in ['dr.', 'patient', 'age', 'date']) and
                    re.search(r'[A-Za-z]', line)):
                    
                    medicine = {
                        'name': line,
                        'dosage': '',
                        'frequency': '',
                        'duration': ''
                    }
                    medicines.append(medicine)
                    
                    if len(medicines) >= 5:  # Limit to prevent noise
                        break
        
        return medicines

    def sanitize_prescription_data(self, data: dict) -> dict:
        """Clean data before Pydantic validation to prevent None errors"""
        # Ensure all string fields have safe defaults
        string_fields = ['patient_name', 'patient_gender', 'doctor_name', 'doctor_license', 
                        'doctor_specialization', 'additional_notes', 'diagnosis']
        for field in string_fields:
            if data.get(field) is None:
                data[field] = ""
        
        # Handle age - convert to int if possible, otherwise None
        age = data.get('patient_age')
        if age is not None:
            try:
                data['patient_age'] = int(age) if age != "" else None
            except (ValueError, TypeError):
                data['patient_age'] = None
        
        # Ensure medications is a list
        if 'medications' not in data or data['medications'] is None:
            data['medications'] = []
        
        # Clean each medication item
        medications = []
        for med in data.get('medications', []):
            if isinstance(med, dict):
                cleaned_med = {
                    'name': med.get('name') or "",
                    'dosage': med.get('dosage') or "",
                    'frequency': med.get('frequency') or "",
                    'duration': med.get('duration') or ""
                }
                medications.append(cleaned_med)
        data['medications'] = medications
        
        return data

    def enhance_medicine_info(self, medicines: List[Dict]) -> List[Medicine]:
        """Enhance medicine information with database lookup"""
        enhanced_medicines = []
        
        for med_dict in medicines:
            medicine = Medicine(
                name=med_dict.get('name', ''),
                dosage=med_dict.get('dosage', ''),
                frequency=med_dict.get('frequency', ''),
                duration=med_dict.get('duration', ''),
                quantity=str(med_dict.get('quantity', 1)),
                available=self._check_availability(med_dict.get('name', ''))
            )
            enhanced_medicines.append(medicine)
        
        return enhanced_medicines

    def _check_availability(self, medicine_name: str) -> bool:
        """Check medicine availability using fuzzy matching"""
        if not medicine_name:
            return True
            
        medicine_lower = medicine_name.lower().strip()
        
        # Direct match
        if medicine_lower in self.medicine_database:
            return self.medicine_database[medicine_lower]['available']
        
        # Fuzzy match
        best_match = process.extractOne(medicine_lower, self.medicine_database.keys(), scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 75:
            return self.medicine_database[best_match[0]]['available']
        
        return True  # Default to available

    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """Main method to analyze prescription image with enhanced doctor/patient detection"""
        try:
            prescription_id = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}{str(uuid.uuid4())[:8]}"
            
            logger.info(f"Starting enhanced analysis for prescription {prescription_id}")
            
            # Preprocess image
            processed_images = self.preprocess_image(image_path)
            if not processed_images:
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(), medicines=[],
                    diagnosis=[], confidence_score=0.0, raw_text="",
                    success=False, error="Failed to preprocess image"
                )
            
            # Extract text
            extracted_text, ocr_confidence = self.extract_text(processed_images)
            if not extracted_text.strip():
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(), medicines=[],
                    diagnosis=[], confidence_score=0.0, raw_text="",
                    success=False, error="No text could be extracted"
                )
            
            # Analyze with Cohere API (or fallback to pattern matching)
            try:
                cohere_result = self.analyze_with_cohere(extracted_text, ocr_confidence)
            except Exception as e:
                logger.warning(f"Cohere API analysis failed: {e}, using pattern-based fallback")
                doctor_info, patient_info = self.extract_doctor_patient_info(extracted_text)
                cohere_result = self._fallback_extraction(extracted_text, ocr_confidence, doctor_info, patient_info)
            
            # Create structured result
            patient = Patient(
                name=cohere_result.get('patient_name', ''),
                age=str(cohere_result.get('patient_age', '')) if cohere_result.get('patient_age') is not None else '',
                gender=cohere_result.get('patient_gender', '')
            )
            
            doctor = Doctor(
                name=cohere_result.get('doctor_name', ''),
                specialization=cohere_result.get('doctor_specialization', ''),
                registration_number=cohere_result.get('doctor_license', '')
            )
            
            # Enhance medicines
            medicines = self.enhance_medicine_info(cohere_result.get('medications', []))
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                ocr_confidence, len(medicines), patient, doctor
            )
            
            logger.info(f"Enhanced analysis completed successfully for {prescription_id}")
            logger.info(f"Doctor: {doctor.name}, Patient: {patient.name}, Medicines: {len(medicines)}")
            
            return AnalysisResult(
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                diagnosis=[cohere_result.get('diagnosis', '')] if cohere_result.get('diagnosis') else [],
                confidence_score=confidence_score,
                raw_text=extracted_text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced analyze_prescription: {e}")
            return AnalysisResult(
                prescription_id=f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}",
                patient=Patient(), doctor=Doctor(), medicines=[],
                diagnosis=[], confidence_score=0.0, raw_text="",
                success=False, error=str(e)
            )

    def _calculate_confidence(self, ocr_confidence: float, medicines_count: int, 
                            patient: Patient, doctor: Doctor) -> float:
        """Calculate overall confidence score with enhanced weighting"""
        weights = {
            'ocr': 0.3, 
            'medicines': 0.3, 
            'patient_info': 0.2, 
            'doctor_info': 0.2
        }
        
        # Medicine score based on count and quality
        medicine_score = 0.0
        if medicines_count > 0:
            medicine_score = min(1.0, medicines_count / 3.0)
            medicine_score = min(1.0, medicine_score + 0.3)  # Bonus for having medicines
        
        # Patient score based on available information
        patient_score = 0.0
        if patient.name: patient_score += 0.5
        if patient.age: patient_score += 0.25
        if patient.gender: patient_score += 0.25
        patient_score = min(1.0, patient_score)
        
        # Doctor score based on available information
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
            # Legacy fields for backward compatibility
            "patient_name": result.patient.name,
            "patient_age": int(result.patient.age) if result.patient.age and result.patient.age.isdigit() else 0,
            "patient_gender": result.patient.gender,
            "doctor_name": result.doctor.name,
            "doctor_license": result.doctor.registration_number
        }

    def train_on_sample_data(self, sample_prescriptions: List[Dict]) -> None:
        """
        Train/improve the analyzer using sample prescription data
        
        Args:
            sample_prescriptions: List of dictionaries containing sample data with expected outputs
        """
        logger.info("Training analyzer on sample prescription data...")
        
        # Update medicine database with new medicines found in samples
        new_medicines = {}
        
        for sample in sample_prescriptions:
            if 'medicines' in sample:
                for medicine in sample['medicines']:
                    medicine_name = medicine.get('name', '').lower().strip()
                    if medicine_name and medicine_name not in self.medicine_database:
                        new_medicines[medicine_name] = {
                            'category': 'unknown',
                            'generic': medicine_name,
                            'available': True
                        }
        
        if new_medicines:
            self.medicine_database.update(new_medicines)
            self._save_medicine_database()
            logger.info(f"Added {len(new_medicines)} new medicines to database")
        
        logger.info("Training completed successfully")