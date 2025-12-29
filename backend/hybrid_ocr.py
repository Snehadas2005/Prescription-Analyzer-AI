"""
Hybrid OCR Engine combining traditional OCR and TrOCR
"""
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    confidence: float
    method: str  # 'easyocr', 'tesseract', 'trocr'
    is_handwritten: bool

class HybridOCREngine:
    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Traditional OCR
        self.easyocr = easyocr.Reader(['en'], gpu=use_gpu)
        
        # TrOCR setup
        print(f"Loading TrOCR on {self.device}...")
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-handwritten'
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten'
        ).to(self.device)
        
        # CPU optimization
        if self.device == "cpu":
            self.trocr_model = torch.quantization.quantize_dynamic(
                self.trocr_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✓ TrOCR quantized for CPU")
        
        self.trocr_model.eval()
        
        # Thresholds
        self.HANDWRITING_THRESHOLD = 0.6
        self.MIN_TEXT_LENGTH = 15
    
    def is_handwritten(self, text: str, confidence: float) -> bool:
        """Heuristic to detect handwriting"""
        # Low confidence suggests handwriting
        if confidence < self.HANDWRITING_THRESHOLD:
            return True
        
        # Very short text suggests OCR failure
        if len(text.strip()) < self.MIN_TEXT_LENGTH:
            return True
        
        # High ratio of special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_ratio / max(len(text), 1) > 0.3:
            return True
        
        return False
    
    def extract_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract with EasyOCR"""
        results = self.easyocr.readtext(image, detail=1)
        if not results:
            return OCRResult("", 0.0, "easyocr", False)
        
        text = " ".join([r[1] for r in results])
        conf = np.mean([r[2] for r in results])
        
        return OCRResult(text, conf, "easyocr", False)
    
    def extract_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract with Tesseract"""
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        conf_scores = [int(c) for c in data['conf'] if int(c) > 0]
        conf = np.mean(conf_scores) / 100 if conf_scores else 0.0
        
        return OCRResult(text, conf, "tesseract", False)
    
    def extract_with_trocr(self, image: np.ndarray) -> OCRResult:
        """Extract with TrOCR (handwriting)"""
        # Convert to PIL
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image).convert('RGB')
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        pixel_values = self.trocr_processor(
            images=pil_img, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)
        
        text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return OCRResult(text.strip(), 0.7, "trocr", True)
    
    def extract_hybrid(self, image: np.ndarray) -> Tuple[str, float, List[str]]:
        """
        Main hybrid extraction
        Returns: (text, confidence, methods_used)
        """
        methods_used = []
        
        # Try traditional OCR first
        easyocr_result = self.extract_with_easyocr(image)
        tesseract_result = self.extract_with_tesseract(image)
        
        # Pick best traditional result
        if easyocr_result.confidence > tesseract_result.confidence:
            best_traditional = easyocr_result
        else:
            best_traditional = tesseract_result
        
        methods_used.append(best_traditional.method)
        
        # Check if we should try TrOCR
        if self.is_handwritten(best_traditional.text, best_traditional.confidence):
            print(f"⚠️ Low confidence ({best_traditional.confidence:.2f}), trying TrOCR...")
            
            trocr_result = self.extract_with_trocr(image)
            methods_used.append('trocr')
            
            # Use TrOCR if it gives more text
            if len(trocr_result.text) > len(best_traditional.text):
                print("✓ TrOCR produced better result")
                return trocr_result.text, trocr_result.confidence, methods_used
        
        return best_traditional.text, best_traditional.confidence, methods_used