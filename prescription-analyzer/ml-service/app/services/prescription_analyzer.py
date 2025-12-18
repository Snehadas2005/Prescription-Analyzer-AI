from __future__ import annotations
import cv2
import numpy as np
import easyocr
import re
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime
import os
import pickle
from PIL import Image
import pytesseract
from fuzzywuzzy import fuzz, process
import cohere
import uuid

# TrOCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("TrOCR dependencies not installed. Run: pip install transformers torch")

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
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None


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
    ocr_methods_used: Optional[List[str]] = field(default_factory=list)


class HybridOCREngine:
    """
    Hybrid OCR Engine that intelligently uses:
    - EasyOCR/Tesseract for printed text
    - TrOCR for handwritten text
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available() if TROCR_AVAILABLE else False
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
        if not TROCR_AVAILABLE:
            logger.warning("TrOCR not available. Install with: pip install transformers torch")
            self.trocr_available = False
            self.trocr_processor = None
            self.trocr_model = None
            return
            
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
                try:
                    self.trocr_model = torch.quantization.quantize_dynamic(
                        self.trocr_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("✓ TrOCR quantized for CPU inference")
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}, continuing without quantization")
            
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
    Enhanced analyzer with hybrid OCR support and self-learning capabilities
    """
    
    def __init__(self, cohere_api_key: str = None, use_gpu: bool = False, 
                 force_api: bool = False, tesseract_path: str = None):
        logger.info("Initializing Enhanced Prescription Analyzer with Hybrid OCR...")
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize Cohere
        self._init_cohere_api(cohere_api_key, force_api)
        
        # Initialize Hybrid OCR Engine
        self.ocr_engine = HybridOCREngine(use_gpu=use_gpu)
        
        # Load medicine database
        self.medicine_database = self._load_medicine_database()
        if not self.medicine_database:
            self.medicine_database = self._create_default_medicine_database()
            self._save_medicine_database()
        
        # Initialize self-learning components
        self.feedback_data = []
        self.training_history = []
        
        # Medical patterns
        self.doctor_patterns = {
            'titles': [
                r'\b(Dr\.?|Doctor|Prof\.?|Professor)\s+([A-Za-z\s\.]+)',
                r'\b(MBBS|MD|MS|DM|MCh|FRCS|MRCP|DNB)\b',
            ],
            'specializations': [
                r'\b(Cardiologist|Neurologist|Orthopedic|Pediatrician|Dermatologist|Gynecologist|Psychiatrist|Radiologist|Anesthesiologist|Pathologist|Oncologist|Urologist|ENT|Ophthalmologist|General\s+Medicine|Internal\s+Medicine|General\s+Physician)\b',
            ],
            'registration': [
                r'\b(Reg\.?\s*No\.?|Registration\s+No\.?|License\s+No\.?|Medical\s+License)\s*:?\s*([A-Z0-9\-/]+)',
                r'\b([A-Z]{2,4}[-/]?\d{4,6})\b'
            ]
        }
        
        self.patient_patterns = {
            'age_indicators': [
                r'\b(Age|age)\s*:?\s*(\d{1,3})\s*(years?|yrs?|Y)?',
                r'\b(\d{1,3})\s*(years?|yrs?|Y)\s*(old|age)?',
            ],
            'gender_indicators': [
                r'\b(Male|Female|M|F|male|female)\b',
                r'\b(Gender|Sex)\s*:?\s*(Male|Female|M|F)',
                r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+',
            ],
            'name_patterns': [
                r'\b(Patient|patient)\s*:?\s*([A-Za-z\s\.]+)',
                r'\b(Name|name)\s*:?\s*([A-Za-z\s\.]+)',
                r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+([A-Za-z\s]+)'
            ]
        }
        
        self.medical_abbreviations = {
            'bd': 'twice daily', 'bid': 'twice daily',
            'tid': 'three times daily', 'qid': 'four times daily',
            'od': 'once daily', 'qd': 'once daily',
            'sos': 'as needed', 'prn': 'as needed',
            'ac': 'before meals', 'pc': 'after meals',
            'hs': 'at bedtime', 'qhs': 'at bedtime',
            'mg': 'milligrams', 'gm': 'grams', 'g': 'grams',
            'ml': 'milliliters', 'cap': 'capsule',
            'tab': 'tablet', 'syp': 'syrup', 'inj': 'injection'
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
                raise ValueError("No Cohere API key provided")
    
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
            logger.info("Medicine database saved")
        except Exception as e:
            logger.warning(f"Failed to save medicine DB: {e}")
    
    def _create_default_medicine_database(self):
        """Create default medicine database"""
        return {
            # Antibiotics
            'augmentin': {'category': 'antibiotic', 'generic': 'amoxicillin + clavulanic acid', 'available': True},
            'amoxicillin': {'category': 'antibiotic', 'generic': 'amoxicillin', 'available': True},
            'azithromycin': {'category': 'antibiotic', 'generic': 'azithromycin', 'available': True},
            'ciprofloxacin': {'category': 'antibiotic', 'generic': 'ciprofloxacin', 'available': True},
            
            # Pain relievers
            'paracetamol': {'category': 'analgesic', 'generic': 'paracetamol', 'available': True},
            'ibuprofen': {'category': 'nsaid', 'generic': 'ibuprofen', 'available': True},
            'diclofenac': {'category': 'nsaid', 'generic': 'diclofenac', 'available': True},
            
            # PPIs
            'omeprazole': {'category': 'ppi', 'generic': 'omeprazole', 'available': True},
            'pantoprazole': {'category': 'ppi', 'generic': 'pantoprazole', 'available': True},
            
            # Antihistamines
            'cetirizine': {'category': 'antihistamine', 'generic': 'cetirizine', 'available': True},
            'loratadine': {'category': 'antihistamine', 'generic': 'loratadine', 'available': True},
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
                if len(match.groups()) >= 2:
                    doctor_info['name'] = match.group(2).strip()
                else:
                    doctor_info['name'] = match.group(1).strip()
        
        # Extract specialization
        for pattern in self.doctor_patterns['specializations']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['specialization']:
                doctor_info['specialization'] = match.group(0).strip()
        
        # Extract registration
        for pattern in self.doctor_patterns['registration']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not doctor_info['registration_number']:
                if len(match.groups()) >= 2:
                    doctor_info['registration_number'] = match.group(2).strip()
                else:
                    doctor_info['registration_number'] = match.group(1).strip()
        
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
                gender_text = match.group(0).lower()
                if 'male' in gender_text and 'female' not in gender_text:
                    patient_info['gender'] = 'Male'
                elif 'female' in gender_text or 'f' == gender_text:
                    patient_info['gender'] = 'Female'
        
        # Extract patient name
        for pattern in self.patient_patterns['name_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not patient_info['name']:
                if len(match.groups()) >= 2:
                    name_candidate = match.group(2).strip()
                    if 2 <= len(name_candidate) <= 50:
                        patient_info['name'] = name_candidate
        
        return doctor_info, patient_info
    
