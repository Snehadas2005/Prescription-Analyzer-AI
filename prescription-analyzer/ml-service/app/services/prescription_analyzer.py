from __future__ import annotations
import cv2
import numpy as np
import easyocr
import re
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from PIL import Image
import pytesseract
from fuzzywuzzy import fuzz, process
import cohere
import uuid

# TrOCR imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix PIL deprecation
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS


@dataclass
class OCRResult:
    """Container for OCR results with metadata"""
    text: str
    confidence: float
    method: str  # 'easyocr', 'tesseract', 'trocr'
    is_handwritten: bool
    bounding_boxes: List[Tuple[int, int, int, int]] = None


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
    ocr_methods_used: List[str] = None


class HybridOCREngine:
    """
    Hybrid OCR Engine that intelligently uses:
    - EasyOCR/Tesseract for printed text
    - TrOCR for handwritten text
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        logger.info(f"Initializing Hybrid OCR Engine on {self.device}")
        
        # Initialize traditional OCR
        self._init_traditional_ocr()
        
        # Initialize TrOCR
        self._init_trocr()
        
        # Thresholds for handwriting detection
        self.HANDWRITING_CONFIDENCE_THRESHOLD = 0.6
        self.MIN_TEXT_LENGTH_THRESHOLD = 15
        
    def _init_traditional_ocr(self):
        """Initialize EasyOCR and Tesseract"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            logger.info("✓ EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
            
    def _init_trocr(self):
        """Initialize TrOCR for handwritten text"""
        try:
            model_name = "microsoft/trocr-base-handwritten"
            
            logger.info(f"Loading TrOCR model: {model_name}")
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Move to device and optimize
            self.trocr_model.to(self.device)
            self.trocr_model.eval()  # Set to evaluation mode
            
            # Enable CPU optimizations if not using GPU
            if not self.use_gpu:
                self.trocr_model = torch.quantization.quantize_dynamic(
                    self.trocr_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✓ TrOCR quantized for CPU inference")
            
            logger.info("✓ TrOCR initialized successfully")
            self.trocr_available = True
            
        except Exception as e:
            logger.error(f"❌ TrOCR initialization failed: {e}")
            logger.warning("Falling back to traditional OCR only")
            self.trocr_available = False
            self.trocr_processor = None
            self.trocr_model = None
    
    def detect_handwriting_regions(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect regions that are likely handwritten
        Returns: List of (region_image, bounding_box) tuples
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        height, width = gray.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 30 or h < 20:
                continue
                
            # Filter regions that are too large (likely whole document)
            if w > width * 0.9 or h > height * 0.9:
                continue
            
            # Extract region with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            region = gray[y1:y2, x1:x2]
            regions.append((region, (x1, y1, x2, y2)))
        
        return regions
    
    def extract_with_trocr(self, image: np.ndarray) -> str:
        """
        Extract text from handwritten image using TrOCR
        """
        if not self.trocr_available:
            return ""
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess for TrOCR
            pixel_values = self.trocr_processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
            
            # Decode
            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return ""
    
    def is_likely_handwritten(self, ocr_result: OCRResult) -> bool:
        """
        Heuristic to determine if text is likely handwritten
        """
        # Low confidence from traditional OCR
        if ocr_result.confidence < self.HANDWRITING_CONFIDENCE_THRESHOLD:
            return True
        
        # Very short text (OCR might have failed)
        if len(ocr_result.text.strip()) < self.MIN_TEXT_LENGTH_THRESHOLD:
            return True
        
        # Check for OCR artifacts (lots of special characters)
        special_char_ratio = sum(1 for c in ocr_result.text if not c.isalnum() and not c.isspace()) / max(len(ocr_result.text), 1)
        if special_char_ratio > 0.3:
            return True
        
        return False
    
    def extract_text_hybrid(self, image: np.ndarray) -> OCRResult:
        """
        Main hybrid extraction method
        Tries traditional OCR first, falls back to TrOCR if needed
        """
        results = []
        
        # Try EasyOCR first
        if self.easyocr_reader:
            try:
                easyocr_results = self.easyocr_reader.readtext(image, detail=1)
                if easyocr_results:
                    text = " ".join([r[1] for r in easyocr_results])
                    confidence = np.mean([r[2] for r in easyocr_results])
                    
                    result = OCRResult(
                        text=text,
                        confidence=confidence,
                        method='easyocr',
                        is_handwritten=False
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Try Tesseract
        try:
            tesseract_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            tesseract_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            conf_scores = [int(c) for c in tesseract_data['conf'] if int(c) > 0]
            if conf_scores and len(tesseract_text.strip()) > 10:
                confidence = np.mean(conf_scores) / 100
                
                result = OCRResult(
                    text=tesseract_text,
                    confidence=confidence,
                    method='tesseract',
                    is_handwritten=False
                )
                results.append(result)
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
        
        # Choose best traditional OCR result
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            
            # Check if we should try TrOCR
            if self.trocr_available and self.is_likely_handwritten(best_result):
                logger.info(f"Low confidence ({best_result.confidence:.2f}), trying TrOCR...")
                
                # Try TrOCR on whole image
                trocr_text = self.extract_with_trocr(image)
                
                if len(trocr_text.strip()) > len(best_result.text.strip()):
                    logger.info(f"✓ TrOCR produced better result")
                    return OCRResult(
                        text=trocr_text,
                        confidence=0.7,  # Assume decent confidence
                        method='trocr',
                        is_handwritten=True
                    )
            
            return best_result
        
        # If traditional OCR completely failed, try TrOCR
        if self.trocr_available:
            logger.info("Traditional OCR failed, trying TrOCR...")
            trocr_text = self.extract_with_trocr(image)
            
            if trocr_text:
                return OCRResult(
                    text=trocr_text,
                    confidence=0.6,
                    method='trocr',
                    is_handwritten=True
                )
        
        # Complete failure
        return OCRResult(
            text="",
            confidence=0.0,
            method='none',
            is_handwritten=False
        )
    
    def extract_text_advanced(self, image: np.ndarray) -> Tuple[str, float, List[str]]:
        """
        Advanced extraction with region-based TrOCR
        Returns: (combined_text, overall_confidence, methods_used)
        """
        methods_used = []
        all_texts = []
        all_confidences = []
        
        # First try traditional OCR on whole image
        traditional_result = self.extract_text_hybrid(image)
        methods_used.append(traditional_result.method)
        
        if traditional_result.confidence > self.HANDWRITING_CONFIDENCE_THRESHOLD:
            # Good traditional OCR result
            return traditional_result.text, traditional_result.confidence, methods_used
        
        # If traditional OCR is poor, try region-based TrOCR
        if self.trocr_available:
            logger.info("Trying region-based TrOCR...")
            regions = self.detect_handwriting_regions(image)
            
            for region_img, bbox in regions[:10]:  # Limit to 10 regions
                trocr_text = self.extract_with_trocr(region_img)
                if trocr_text:
                    all_texts.append(trocr_text)
                    all_confidences.append(0.7)
                    if 'trocr' not in methods_used:
                        methods_used.append('trocr')
        
        # Combine results
        if all_texts:
            combined_text = " ".join(all_texts)
            overall_confidence = np.mean(all_confidences)
        else:
            combined_text = traditional_result.text
            overall_confidence = traditional_result.confidence
        
        return combined_text, overall_confidence, methods_used


class EnhancedPrescriptionAnalyzer:
    """
    Enhanced analyzer with hybrid OCR support
    """
    
    def __init__(self, cohere_api_key: str = None, use_gpu: bool = False, force_api: bool = False):
        logger.info("Initializing Enhanced Prescription Analyzer with Hybrid OCR...")
        
        # Initialize Cohere
        self._init_cohere_api(cohere_api_key, force_api)
        
        # Initialize Hybrid OCR Engine
        self.ocr_engine = HybridOCREngine(use_gpu=use_gpu)
        
        # Load medicine database
        self.medicine_database = self._load_medicine_database()
        if not self.medicine_database:
            self.medicine_database = self._create_default_medicine_database()
            self._save_medicine_database()
        
        # Medical patterns (same as before)
        self.doctor_patterns = {
            'titles': [
                r'\b(Dr\.?|Doctor|Prof\.?)\s+([A-Za-z\s\.]+)',
                r'\b(MBBS|MD|MS|DM|MCh)\b',
            ],
            'specializations': [
                r'\b(Cardiologist|Neurologist|Pediatrician|Dermatologist|General\s+Physician)\b',
            ],
            'registration': [
                r'\b(Reg\.?\s*No\.?|License)\s*:?\s*([A-Z0-9]+)',
            ]
        }
        
        self.patient_patterns = {
            'age_indicators': [
                r'\b(Age|age)\s*:?\s*(\d{1,3})',
                r'\b(\d{1,3})\s*(years?|yrs?|Y)',
            ],
            'gender_indicators': [
                r'\b(Male|Female|M|F)\b',
                r'\b(Gender|Sex)\s*:?\s*(Male|Female|M|F)',
            ],
            'name_patterns': [
                r'\b(Patient|Name)\s*:?\s*([A-Za-z\s\.]+)',
                r'\b(Mr\.?|Mrs\.?|Ms\.?)\s+([A-Za-z\s]+)'
            ]
        }
        
        self.medical_abbreviations = {
            'bd': 'twice daily', 'tid': 'three times daily',
            'qid': 'four times daily', 'od': 'once daily',
            'mg': 'milligrams', 'ml': 'milliliters',
            'tab': 'tablet', 'cap': 'capsule',
        }
    
    def _init_cohere_api(self, api_key: str, force_api: bool):
        """Initialize Cohere API"""
        key_from_file = None
        try:
            from integration.keys import COHERE_API_KEY
            key_from_file = COHERE_API_KEY
        except ImportError:
            pass
        
        final_key = api_key or os.getenv("COHERE_API_KEY") or key_from_file
        
        self.co = None
        if final_key:
            try:
                self.co = cohere.Client(final_key)
                logger.info("✓ Cohere API initialized")
            except Exception as e:
                logger.error(f"Cohere init failed: {e}")
                if force_api:
                    raise
        else:
            if force_api:
                raise ValueError("No Cohere API key")
    
    def _load_medicine_database(self):
        """Load medicine database"""
        try:
            with open("medicine_database.pkl", "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return None
    
    def _save_medicine_database(self):
        """Save medicine database"""
        try:
            with open("medicine_database.pkl", "wb") as f:
                pickle.dump(self.medicine_database, f)
        except Exception as e:
            logger.warning(f"Failed to save medicine DB: {e}")
    
    def _create_default_medicine_database(self):
        """Create default medicine database"""
        return {
            'paracetamol': {'category': 'analgesic', 'available': True},
            'ibuprofen': {'category': 'nsaid', 'available': True},
            'amoxicillin': {'category': 'antibiotic', 'available': True},
            'azithromycin': {'category': 'antibiotic', 'available': True},
            'omeprazole': {'category': 'ppi', 'available': True},
            'cetirizine': {'category': 'antihistamine', 'available': True},
        }
    
    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """Preprocess image for OCR"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if too small
        height, width = gray.shape
        if height < 800 or width < 600:
            scale = max(800/height, 600/width)
            new_w, new_h = int(width * scale), int(height * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        processed = []
        
        # Method 1: CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            processed.append(thresh)
        except Exception as e:
            logger.warning(f"CLAHE preprocessing failed: {e}")
        
        # Method 2: Otsu
        try:
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed.append(otsu)
        except Exception as e:
            logger.warning(f"Otsu preprocessing failed: {e}")
        
        return processed if processed else [gray]
    
    def extract_text(self, processed_images: List[np.ndarray]) -> Tuple[str, float, List[str]]:
        """
        Extract text using hybrid OCR
        Returns: (text, confidence, methods_used)
        """
        best_text = ""
        best_confidence = 0.0
        all_methods = []
        
        for image in processed_images:
            text, confidence, methods = self.ocr_engine.extract_text_advanced(image)
            
            all_methods.extend(methods)
            
            if confidence > best_confidence:
                best_text = text
                best_confidence = confidence
        
        # Deduplicate methods
        unique_methods = list(set(all_methods))
        
        logger.info(f"Best OCR result: {best_confidence:.2f} using {unique_methods}")
        
        return best_text, best_confidence, unique_methods
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Common OCR fixes
        fixes = {
            r'rng': 'mg', r'\.mg': ' mg',
            r'Tab\b': 'Tab', r'Cap\b': 'Cap',
            r'rnl': 'ml', r'\b0\b': 'O',
        }
        
        for pattern, repl in fixes.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_doctor_patient_info(self, text: str) -> Tuple[Dict, Dict]:
        """Extract doctor and patient info using patterns"""
        doctor_info = {'name': '', 'specialization': '', 'registration_number': ''}
        patient_info = {'name': '', 'age': '', 'gender': ''}
        
        # Extract doctor name
        for pattern in self.doctor_patterns['titles']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['name']:
                doctor_info['name'] = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
        
        # Extract specialization
        for pattern in self.doctor_patterns['specializations']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['specialization']:
                doctor_info['specialization'] = match.group(0).strip()
        
        # Extract patient age
        for pattern in self.patient_patterns['age_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['age']:
                for group in match.groups():
                    if group and group.isdigit():
                        patient_info['age'] = group
                        break
        
        # Extract patient gender
        for pattern in self.patient_patterns['gender_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['gender']:
                gender = match.group(0).lower()
                if 'male' in gender or 'm' in gender:
                    patient_info['gender'] = 'Male' if 'female' not in gender else 'Female'
        
        # Extract patient name
        for pattern in self.patient_patterns['name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['name']:
                if len(match.groups()) >= 2:
                    patient_info['name'] = match.group(2).strip()
        
        return doctor_info, patient_info
    
    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """Main analysis method"""
        try:
            prescription_id = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting analysis for {prescription_id}")
            
            # Preprocess
            processed_images = self.preprocess_image(image_path)
            if not processed_images:
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(),
                    medicines=[], diagnosis=[],
                    confidence_score=0.0, raw_text="",
                    success=False, error="Preprocessing failed",
                    ocr_methods_used=[]
                )
            
            # Extract text with hybrid OCR
            extracted_text, ocr_confidence, methods_used = self.extract_text(processed_images)
            
            if not extracted_text.strip():
                return AnalysisResult(
                    prescription_id=prescription_id,
                    patient=Patient(), doctor=Doctor(),
                    medicines=[], diagnosis=[],
                    confidence_score=0.0, raw_text="",
                    success=False, error="No text extracted",
                    ocr_methods_used=methods_used
                )
            
            # Clean text
            cleaned_text = self._clean_text(extracted_text)
            
            # Extract info
            doctor_info, patient_info = self.extract_doctor_patient_info(cleaned_text)
            
            # Create result
            patient = Patient(
                name=patient_info.get('name', ''),
                age=patient_info.get('age', ''),
                gender=patient_info.get('gender', '')
            )
            
            doctor = Doctor(
                name=doctor_info.get('name', ''),
                specialization=doctor_info.get('specialization', ''),
                registration_number=doctor_info.get('registration_number', '')
            )
            
            # Extract medicines (simplified)
            medicines = self._extract_medicines_simple(cleaned_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(ocr_confidence, len(medicines), patient, doctor)
            
            logger.info(f"✓ Analysis complete: {confidence:.2%} confidence")
            logger.info(f"  OCR methods: {methods_used}")
            logger.info(f"  Doctor: {doctor.name}")
            logger.info(f"  Patient: {patient.name}")
            logger.info(f"  Medicines: {len(medicines)}")
            
            return AnalysisResult(
                prescription_id=prescription_id,
                patient=patient,
                doctor=doctor,
                medicines=medicines,
                diagnosis=[],
                confidence_score=confidence,
                raw_text=cleaned_text,
                success=True,
                ocr_methods_used=methods_used
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                prescription_id=f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}",
                patient=Patient(), doctor=Doctor(),
                medicines=[], diagnosis=[],
                confidence_score=0.0, raw_text="",
                success=False, error=str(e),
                ocr_methods_used=[]
            )
    
    def _extract_medicines_simple(self, text: str) -> List[Medicine]:
        """Simple medicine extraction"""
        medicines = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Look for medicine patterns
            medicine_match = re.search(r'(\w+(?:\s+\w+)?)\s+(\d+\s*(?:mg|ml))', line, re.IGNORECASE)
            
            if medicine_match:
                name = medicine_match.group(1)
                dosage = medicine_match.group(2)
                
                medicine = Medicine(
                    name=name,
                    dosage=dosage,
                    quantity="1",
                    frequency="As directed",
                    duration="As prescribed",
                    instructions="",
                    available=self._check_availability(name)
                )
                medicines.append(medicine)
        
        return medicines
    
    def _check_availability(self, medicine_name: str) -> bool:
        """Check medicine availability"""
        if not medicine_name:
            return True
        
        name_lower = medicine_name.lower().strip()
        
        if name_lower in self.medicine_database:
            return self.medicine_database[name_lower]['available']
        
        # Fuzzy match
        best_match = process.extractOne(name_lower, self.medicine_database.keys(), scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 75:
            return self.medicine_database[best_match[0]]['available']
        
        return True
    
    def _calculate_confidence(self, ocr_conf: float, med_count: int, 
                            patient: Patient, doctor: Doctor) -> float:
        """Calculate overall confidence"""
        weights = {'ocr': 0.4, 'medicines': 0.2, 'patient': 0.2, 'doctor': 0.2}
        
        med_score = min(1.0, med_count / 3.0)
        patient_score = sum([bool(patient.name), bool(patient.age), bool(patient.gender)]) / 3
        doctor_score = sum([bool(doctor.name), bool(doctor.specialization)]) / 2
        
        return (
            weights['ocr'] * ocr_conf +
            weights['medicines'] * med_score +
            weights['patient'] * patient_score +
            weights['doctor'] * doctor_score
        )
    
    def to_json(self, result: AnalysisResult) -> Dict:
        """Convert result to JSON"""
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
            "raw_text": result.raw_text,
            "message": "Analysis completed successfully" if result.success else result.error,
            "error": result.error if not result.success else "",
            "ocr_methods_used": result.ocr_methods_used or [],
            # Legacy fields for backward compatibility
            "patient_name": result.patient.name,
            "patient_age": int(result.patient.age) if result.patient.age and result.patient.age.isdigit() else 0,
            "patient_gender": result.patient.gender,
            "doctor_name": result.doctor.name,
            "doctor_license": result.doctor.registration_number
        }


# ============================================================================
# USAGE EXAMPLE & TESTING
# ============================================================================

def main():
    """Example usage of the Enhanced Prescription Analyzer with TrOCR"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prescription_analyzer.py <image_path>")
        print("\nExample:")
        print("  python prescription_analyzer.py sample_prescription.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("="*80)
    print("ENHANCED PRESCRIPTION ANALYZER WITH TrOCR")
    print("="*80)
    print(f"\nAnalyzing: {image_path}")
    print("-"*80)
    
    try:
        # Initialize analyzer
        print("\n1. Initializing analyzer...")
        analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=os.getenv('COHERE_API_KEY'),
            use_gpu=False,  # Set to True if you have GPU
            force_api=False
        )
        
        # Analyze prescription
        print("\n2. Analyzing prescription...")
        result = analyzer.analyze_prescription(image_path)
        
        # Display results
        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        
        if result.success:
            print(f"\n✓ SUCCESS")
            print(f"\nPrescription ID: {result.prescription_id}")
            print(f"Confidence Score: {result.confidence_score:.2%}")
            print(f"OCR Methods Used: {', '.join(result.ocr_methods_used)}")
            
            # Patient Info
            print(f"\n{'PATIENT INFORMATION':-^80}")
            print(f"  Name:   {result.patient.name or 'Not detected'}")
            print(f"  Age:    {result.patient.age or 'Not detected'}")
            print(f"  Gender: {result.patient.gender or 'Not detected'}")
            
            # Doctor Info
            print(f"\n{'DOCTOR INFORMATION':-^80}")
            print(f"  Name:           {result.doctor.name or 'Not detected'}")
            print(f"  Specialization: {result.doctor.specialization or 'Not detected'}")
            print(f"  Registration:   {result.doctor.registration_number or 'Not detected'}")
            
            # Medicines
            print(f"\n{'MEDICINES':-^80}")
            if result.medicines:
                for i, med in enumerate(result.medicines, 1):
                    print(f"\n  Medicine {i}:")
                    print(f"    Name:      {med.name}")
                    print(f"    Dosage:    {med.dosage}")
                    print(f"    Frequency: {med.frequency}")
                    print(f"    Duration:  {med.duration}")
                    print(f"    Available: {'✓' if med.available else '✗'}")
            else:
                print("  No medicines detected")
            
            # Raw Text
            print(f"\n{'RAW EXTRACTED TEXT':-^80}")
            print(f"{result.raw_text[:500]}..." if len(result.raw_text) > 500 else result.raw_text)
            
            # JSON Output
            print(f"\n{'JSON OUTPUT':-^80}")
            json_output = analyzer.to_json(result)
            print(json.dumps(json_output, indent=2, default=str))
            
        else:
            print(f"\n✗ FAILED")
            print(f"Error: {result.error}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()