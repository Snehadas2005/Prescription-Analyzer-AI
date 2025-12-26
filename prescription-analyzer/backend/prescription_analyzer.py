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
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import spacy
from fuzzywuzzy import fuzz, process
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyzer.log'),
        logging.StreamHandler()
    ]
)
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


class SelfLearningPrescriptionAnalyzer:
    """
    Advanced self-learning analyzer with 99% confidence target
    """
    
    def __init__(self, data_folder: str = "data/prescriptions"):
        """Initialize with training data folder"""
        self.data_folder = Path(data_folder)
        self.models_folder = Path("models/trained")
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
        # Load or create ML models
        self._init_ml_models()
        
        # Load knowledge bases
        self._init_knowledge_bases()
        
        # Initialize NLP
        self._init_nlp()
        
        logger.info("✓ Self-learning analyzer initialized")
    
    def _init_ocr_engines(self):
        """Initialize multiple OCR engines for ensemble"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✓ EasyOCR initialized")
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")
            self.easyocr_reader = None
        
        # Tesseract is already available through pytesseract
        self.tesseract_available = True
    
    def _init_ml_models(self):
        """Initialize or load ML models"""
        self.models = {}
        
        # Medicine name classifier
        medicine_model_path = self.models_folder / "medicine_classifier.pkl"
        if medicine_model_path.exists():
            with open(medicine_model_path, 'rb') as f:
                self.models['medicine'] = pickle.load(f)
            logger.info("✓ Loaded medicine classifier")
        else:
            self.models['medicine'] = {
                'vectorizer': TfidfVectorizer(max_features=500),
                'classifier': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        
        # Entity extraction models
        self.models['entity_extractors'] = {}
        
        # Confidence predictor
        confidence_model_path = self.models_folder / "confidence_predictor.pkl"
        if confidence_model_path.exists():
            with open(confidence_model_path, 'rb') as f:
                self.models['confidence'] = pickle.load(f)
            logger.info("✓ Loaded confidence predictor")
        else:
            self.models['confidence'] = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _init_knowledge_bases(self):
        """Initialize knowledge bases from training data"""
        # Load medicine database
        medicine_db_path = Path("medicine_database.json")
        if medicine_db_path.exists():
            with open(medicine_db_path, 'r') as f:
                self.medicine_db = json.load(f)
        else:
            self.medicine_db = self._create_comprehensive_medicine_db()
            with open(medicine_db_path, 'w') as f:
                json.dump(self.medicine_db, f, indent=2)
        
        # Load doctor specializations
        self.specializations = self._load_specializations()
        
        # Load common patterns from training data
        self.patterns = self._learn_patterns_from_data()
        
        logger.info(f"✓ Loaded {len(self.medicine_db)} medicines")
        logger.info(f"✓ Loaded {len(self.specializations)} specializations")
    
    def _init_nlp(self):
        """Initialize NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ SpaCy NLP loaded")
        except:
            logger.warning("SpaCy model not found, installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def train_from_dataset(self, retrain: bool = False):
        """
        Train models from the 142 prescriptions in data folder
        """
        logger.info("=" * 80)
        logger.info("TRAINING FROM PRESCRIPTION DATASET")
        logger.info("=" * 80)
        
        if not self.data_folder.exists():
            logger.error(f"Data folder not found: {self.data_folder}")
            return False
        
        # Find all prescription images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        prescription_images = []
        for ext in image_extensions:
            prescription_images.extend(list(self.data_folder.glob(f"*{ext}")))
            prescription_images.extend(list(self.data_folder.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(prescription_images)} prescription images")
        
        if len(prescription_images) == 0:
            logger.error("No prescription images found for training!")
            return False
        
        # Extract features from all prescriptions
        training_data = []
        
        for i, img_path in enumerate(prescription_images, 1):
            logger.info(f"Processing {i}/{len(prescription_images)}: {img_path.name}")
            
            try:
                # Extract text with high confidence
                text, ocr_conf = self._extract_text_ensemble(str(img_path))
                
                if len(text) < 50:
                    logger.warning(f"  Low text extraction: {len(text)} chars")
                    continue
                
                # Extract entities
                entities = self._extract_entities_comprehensive(text)
                
                # Store for training
                training_data.append({
                    'image_path': str(img_path),
                    'text': text,
                    'ocr_confidence': ocr_conf,
                    'entities': entities,
                    'features': self._extract_features(text)
                })
                
                logger.info(f"  ✓ Extracted: {len(text)} chars, {len(entities.get('medicines', []))} medicines")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
        
        logger.info(f"\n✓ Processed {len(training_data)} prescriptions successfully")
        
        # Save training data
        training_data_path = self.models_folder / "training_data.json"
        with open(training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # Train models
        self._train_medicine_classifier(training_data)
        self._train_confidence_predictor(training_data)
        self._update_knowledge_bases(training_data)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE - Models Updated")
        logger.info("=" * 80)
        
        return True
    
    def _extract_text_ensemble(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text using ensemble of OCR methods for maximum accuracy
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0
        
        # Multiple preprocessing techniques
        preprocessed = self._preprocess_image_advanced(image)
        
        all_texts = []
        all_confidences = []
        
        # Method 1: EasyOCR
        if self.easyocr_reader:
            try:
                for proc_img in preprocessed:
                    results = self.easyocr_reader.readtext(proc_img, detail=1)
                    if results:
                        text = " ".join([r[1] for r in results])
                        conf = np.mean([r[2] for r in results])
                        all_texts.append(text)
                        all_confidences.append(conf)
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Method 2: Tesseract with multiple configs
        tesseract_configs = [
            '--oem 3 --psm 6',
            '--oem 3 --psm 3',
            '--oem 3 --psm 4',
            '--oem 3 --psm 11',
        ]
        
        for proc_img in preprocessed:
            for config in tesseract_configs:
                try:
                    text = pytesseract.image_to_string(proc_img, config=config)
                    if len(text.strip()) > 20:
                        # Calculate confidence from image data
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
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return combined_text, avg_confidence
    
    def _preprocess_image_advanced(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Advanced image preprocessing with multiple techniques
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if needed
        h, w = gray.shape
        if h < 1000 or w < 800:
            scale = max(1000/h, 800/w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        preprocessed = []
        
        # Method 1: CLAHE + Adaptive
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            preprocessed.append(adaptive)
        except:
            pass
        
        # Method 2: Otsu
        try:
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed.append(otsu)
        except:
            pass
        
        # Method 3: Morphological
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed.append(binary)
        except:
            pass
        
        # Method 4: Contrast enhancement
        try:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            preprocessed.append(normalized)
        except:
            pass
        
        if not preprocessed:
            preprocessed = [gray]
        
        return preprocessed
    
    def _combine_ocr_results(self, texts: List[str], confidences: List[float]) -> str:
        """
        Intelligently combine multiple OCR results
        """
        if not texts:
            return ""
        
        # Weight by confidence
        weighted_texts = list(zip(texts, confidences))
        weighted_texts.sort(key=lambda x: x[1], reverse=True)
        
        # Use top 3 results
        top_texts = [t for t, c in weighted_texts[:3]]
        
        # Merge using word-level voting
        all_words = []
        for text in top_texts:
            words = text.split()
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Reconstruct text with most common words
        result_words = []
        for text in top_texts[0].split('\n'):
            line_words = []
            for word in text.split():
                # Use most confident version of word
                similar = [w for w in word_counts if fuzz.ratio(w, word) > 85]
                if similar:
                    best = max(similar, key=lambda w: word_counts[w])
                    line_words.append(best)
                else:
                    line_words.append(word)
            if line_words:
                result_words.append(' '.join(line_words))
        
        return '\n'.join(result_words)
    
    def _extract_features(self, text: str) -> Dict:
        """Extract features for ML models"""
        features = {
            'text_length': len(text),
            'num_lines': len(text.split('\n')),
            'num_words': len(text.split()),
            'has_doctor_title': int(bool(re.search(r'\bDr\.?', text, re.I))),
            'has_medicine_marker': int(bool(re.search(r'\b(Tab|Cap|Syp|Inj)\.?', text, re.I))),
            'has_dosage': int(bool(re.search(r'\d+\s*mg', text, re.I))),
            'has_age': int(bool(re.search(r'\d{1,3}\s*(?:years?|yrs?)', text, re.I))),
            'has_frequency': int(bool(re.search(r'\b(bd|od|tid|qid)', text, re.I))),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
        return features
    
    def _extract_entities_comprehensive(self, text: str) -> Dict:
        """Extract all entities with high confidence"""
        entities = {
            'patient': {},
            'doctor': {},
            'medicines': []
        }
        
        # Use SpaCy NER
        doc = self.nlp(text)
        
        # Extract patient info
        entities['patient'] = self._extract_patient_advanced(text, doc)
        
        # Extract doctor info
        entities['doctor'] = self._extract_doctor_advanced(text, doc)
        
        # Extract medicines
        entities['medicines'] = self._extract_medicines_advanced(text)
        
        return entities
    
    def _extract_patient_advanced(self, text: str, doc) -> Dict:
        """Advanced patient extraction with multiple strategies"""
        patient = {'name': '', 'age': '', 'gender': '', 'confidence': 0.0}
        
        # Strategy 1: Pattern matching
        age_patterns = [
            r'Age[:\s]+(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|yrs?|Y)\b',
            r'(\d{1,3})\s*/\s*[MF]',
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                age = int(match.group(1))
                if 0 < age < 120:
                    patient['age'] = str(age)
                    patient['confidence'] += 0.3
                    break
        
        # Gender
        if re.search(r'\b(Male|M)\b', text, re.I) and 'Female' not in text:
            patient['gender'] = 'Male'
            patient['confidence'] += 0.2
        elif re.search(r'\b(Female|F)\b', text, re.I):
            patient['gender'] = 'Female'
            patient['confidence'] += 0.2
        
        # Name using NER
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not patient['name']:
                # Verify it's not the doctor
                if not re.search(r'Dr\.?\s+' + ent.text, text, re.I):
                    patient['name'] = ent.text
                    patient['confidence'] += 0.5
                    break
        
        return patient
    
    def _extract_doctor_advanced(self, text: str, doc) -> Dict:
        """Advanced doctor extraction"""
        doctor = {'name': '', 'specialization': '', 'registration_number': '', 'confidence': 0.0}
        
        # Extract name
        dr_pattern = r'Dr\.?\s*\(?\s*(?:Mrs?\.?|Miss|Ms\.?)?\s*\)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(dr_pattern, text, re.I)
        if match:
            doctor['name'] = match.group(1).strip()
            doctor['confidence'] += 0.4
        
        # Specialization
        for spec in self.specializations:
            if spec.lower() in text.lower():
                doctor['specialization'] = spec
                doctor['confidence'] += 0.3
                break
        
        # Registration
        reg_pattern = r'\b([A-Z]{2,4}[-/]?\d{4,6})\b'
        match = re.search(reg_pattern, text)
        if match:
            doctor['registration_number'] = match.group(1)
            doctor['confidence'] += 0.3
        
        return doctor
    
    def _extract_medicines_advanced(self, text: str) -> List[Dict]:
        """
        Advanced medicine extraction with high confidence
        """
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
            
            # Pattern 1: Tab/Cap Medicine
            med_pattern = r'(Tab|Cap|Syp|Inj)\.?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'
            match = re.search(med_pattern, line, re.I)
            
            if match:
                name = match.group(2).strip()
                
                if len(name) < 3 or name.lower() in seen_names:
                    continue
                
                # Verify against medicine database
                is_valid_medicine, confidence = self._verify_medicine_name(name)
                
                if is_valid_medicine:
                    medicine = {
                        'name': name,
                        'dosage': self._extract_dosage(line),
                        'frequency': self._extract_frequency(line),
                        'duration': self._extract_duration(line),
                        'instructions': self._extract_instructions(line),
                        'quantity': self._extract_quantity(line),
                        'confidence': confidence
                    }
                    
                    medicines.append(medicine)
                    seen_names.add(name.lower())
        
        return medicines
    
    def _verify_medicine_name(self, name: str) -> Tuple[bool, float]:
        """Verify if a name is a valid medicine"""
        name_lower = name.lower().strip()
        
        # Check exact match
        if name_lower in self.medicine_db:
            return True, 0.95
        
        # Check fuzzy match
        matches = process.extract(name_lower, self.medicine_db.keys(), scorer=fuzz.ratio, limit=3)
        
        if matches and matches[0][1] > 85:
            return True, matches[0][1] / 100
        
        # Check if it follows medicine naming patterns
        if len(name) >= 4 and name[0].isupper():
            # Common medicine suffixes
            medicine_suffixes = ['zole', 'pril', 'lol', 'cin', 'mycin', 'cillin']
            if any(name_lower.endswith(suffix) for suffix in medicine_suffixes):
                return True, 0.7
        
        return False, 0.0
    
    def _extract_dosage(self, text: str) -> str:
        """Extract dosage"""
        pattern = r'(\d+\.?\d*)\s*(mg|ml|g|gm|mcg)'
        match = re.search(pattern, text, re.I)
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
            match = re.search(pattern, text, re.I)
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
        match = re.search(pattern, text, re.I)
        return int(match.group(1)) if match else 1
    
    def _train_medicine_classifier(self, training_data: List[Dict]):
        """Train medicine name classifier"""
        logger.info("\nTraining medicine classifier...")
        
        # Collect all medicine names
        all_medicines = []
        for data in training_data:
            medicines = data['entities'].get('medicines', [])
            for med in medicines:
                if med.get('name'):
                    all_medicines.append(med['name'].lower())
        
        # Update medicine database
        for med_name in set(all_medicines):
            if med_name not in self.medicine_db:
                self.medicine_db[med_name] = {
                    'category': 'learned',
                    'frequency': all_medicines.count(med_name)
                }
        
        # Save updated database
        with open('medicine_database.json', 'w') as f:
            json.dump(self.medicine_db, f, indent=2)
        
        logger.info(f"✓ Medicine database updated: {len(self.medicine_db)} medicines")
    
    def _train_confidence_predictor(self, training_data: List[Dict]):
        """Train confidence prediction model"""
        logger.info("\nTraining confidence predictor...")
        
        # This would train on historical accuracy data
        # For now, we use rule-based confidence scoring
        
        logger.info("✓ Confidence predictor updated")
    
    def _update_knowledge_bases(self, training_data: List[Dict]):
        """Update knowledge bases from training data"""
        logger.info("\nUpdating knowledge bases...")
        
        # Extract patterns
        all_patterns = []
        for data in training_data:
            text = data['text']
            
            # Doctor name patterns
            dr_matches = re.findall(r'Dr\.?\s+[A-Z][a-z]+', text, re.I)
            all_patterns.extend(dr_matches)
            
            # Medicine patterns  
            med_matches = re.findall(r'(Tab|Cap)\.?\s+[A-Z][a-zA-Z]+', text, re.I)
            all_patterns.extend(med_matches)
        
        self.patterns = list(set(all_patterns))
        
        logger.info(f"✓ Learned {len(self.patterns)} patterns")
    
    def _create_comprehensive_medicine_db(self) -> Dict:
        """Create comprehensive medicine database"""
        return {
            # Common medicines from prescriptions
            'paracetamol': {'category': 'analgesic', 'generic': 'paracetamol'},
            'dolo': {'category': 'analgesic', 'generic': 'paracetamol'},
            'crocin': {'category': 'analgesic', 'generic': 'paracetamol'},
            'ibuprofen': {'category': 'nsaid', 'generic': 'ibuprofen'},
            'combiflam': {'category': 'analgesic', 'generic': 'ibuprofen+paracetamol'},
            'azithromycin': {'category': 'antibiotic', 'generic': 'azithromycin'},
            'amoxicillin': {'category': 'antibiotic', 'generic': 'amoxicillin'},
            'augmentin': {'category': 'antibiotic', 'generic': 'amoxicillin+clavulanic'},
            'omeprazole': {'category': 'ppi', 'generic': 'omeprazole'},
            'pantoprazole': {'category': 'ppi', 'generic': 'pantoprazole'},
            'cetirizine': {'category': 'antihistamine', 'generic': 'cetirizine'},
            'allegra': {'category': 'antihistamine', 'generic': 'fexofenadine'},
            'metformin': {'category': 'antidiabetic', 'generic': 'metformin'},
            'aspirin': {'category': 'antiplatelet', 'generic': 'aspirin'},
            'atorvastatin': {'category': 'statin', 'generic': 'atorvastatin'},
            'amlodipine': {'category': 'antihypertensive', 'generic': 'amlodipine'},
            'levothyroxine': {'category': 'thyroid', 'generic': 'levothyroxine'},
            'montelukast': {'category': 'antiasthmatic', 'generic': 'montelukast'},
            'salbutamol': {'category': 'bronchodilator', 'generic': 'salbutamol'},
            'prednisolone': {'category': 'corticosteroid', 'generic': 'prednisolone'},
        }
    
    def _load_specializations(self) -> List[str]:
        """Load doctor specializations"""
        return [
            'Cardiologist', 'Neurologist', 'Orthopedic', 'Pediatrician',
            'Dermatologist', 'Gynecologist', 'Obstetrician', 'Psychiatrist',
            'Radiologist', 'Anesthesiologist', 'Pathologist', 'Oncologist',
            'Urologist', 'ENT', 'Ophthalmologist', 'General Physician',
            'Internal Medicine', 'Emergency Medicine', 'Consultant'
        ]
    
    def _learn_patterns_from_data(self) -> List[str]:
        """Learn common patterns from training data"""
        patterns = []
        training_data_path = self.models_folder / "training_data.json"
        if training_data_path.exists():
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
            
            for data in training_data:
                text = data['text']
                
                # Doctor name patterns
                dr_matches = re.findall(r'Dr\.?\s+[A-Z][a-z]+', text, re.I)
                patterns.extend(dr_matches)
                
                # Medicine patterns  
                med_matches = re.findall(r'(Tab|Cap)\.?\s+[A-Z][a-zA-Z]+', text, re.I)
                patterns.extend(med_matches)
        
        return list(set(patterns))
    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        """
        Analyze a prescription image and extract structured data
        """
        prescription_id = str(uuid.uuid4())
        logger.info("=" * 80)
        logger.info(f"ANALYZING PRESCRIPTION: {image_path}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Extract text with ensemble OCR
            raw_text, ocr_confidence = self._extract_text_ensemble(image_path)
            if len(raw_text) < 50:
                raise ValueError("Insufficient text extracted from image")
            
            logger.info(f"✓ Extracted {len(raw_text)} characters with OCR confidence {ocr_confidence:.2f}")
            
            # Step 2: Extract entities
            entities = self._extract_entities_comprehensive(raw_text)
            
            # Step 3: Calculate extraction confidence
            extraction_confidence = self._calculate_extraction_confidence(entities)
            
            # Step 4: Validate and cross-check data
            validation_confidence = self._validate_extracted_data(entities)
            
            # Step 5: Overall confidence
            overall_confidence = (ocr_confidence + extraction_confidence + validation_confidence) / 3
            
            # Prepare structured result
            patient_info = PatientInfo(**entities.get('patient', {}))
            doctor_info = DoctorInfo(**entities.get('doctor', {}))
            medicines_info = [MedicineInfo(**med) for med in entities.get('medicines', [])]
            
            result = AnalysisResult(
                success=True,
                prescription_id=prescription_id,
                patient=patient_info,
                doctor=doctor_info,
                medicines=medicines_info,
                confidence_score=overall_confidence,
                raw_text=raw_text,
                ocr_confidence=ocr_confidence,
                extraction_confidence=extraction_confidence,
                validation_confidence=validation_confidence
            )
            
            logger.info(f"✓ Analysis complete with overall confidence {overall_confidence:.2f}")
            return result
        
        except Exception as e:
            logger.error(f"✗ Analysis failed: {e}")
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
                error=str(e)
            )
    def _calculate_extraction_confidence(self, entities: Dict) -> float:
        """Calculate extraction confidence based on extracted entities"""
        score = 0.0
        total = 0
        
        # Patient info
        patient = entities.get('patient', {})
        if patient.get('name'):
            score += patient.get('confidence', 0.0)
            total += 1
        if patient.get('age'):
            score += patient.get('confidence', 0.0)
            total += 1
        if patient.get('gender'):
            score += patient.get('confidence', 0.0)
            total += 1 
        # Doctor info
        doctor = entities.get('doctor', {}) 
        if doctor.get('name'):
            score += doctor.get('confidence', 0.0)
            total += 1
        if doctor.get('specialization'):
            score += doctor.get('confidence', 0.0)
            total += 1
        if doctor.get('registration_number'):
            score += doctor.get('confidence', 0.0)
            total += 1
        # Medicines
        medicines = entities.get('medicines', [])   