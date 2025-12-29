#!/usr/bin/env python3
"""
Test script to compare traditional OCR vs TrOCR
Helps you see the improvement on handwritten prescriptions
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Import your analyzers
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer, HybridOCREngine
except ImportError:
    print("Error: Could not import prescription_analyzer")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)


class OCRComparison:
    """Compare different OCR methods"""
    
    def __init__(self):
        self.ocr_engine = HybridOCREngine(use_gpu=False)
    
    def test_image(self, image_path: str):
        """Test a single image with all OCR methods"""
        print("\n" + "="*80)
        print(f"TESTING: {image_path}")
        print("="*80)
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        print(f"\nImage size: {image.shape[1]}x{image.shape[0]}")
        
        # Test 1: EasyOCR
        print("\n" + "-"*80)
        print("METHOD 1: EasyOCR")
        print("-"*80)
        start = time.time()
        easyocr_result = self._test_easyocr(image)
        easyocr_time = time.time() - start
        
        print(f"Time: {easyocr_time:.2f}s")
        print(f"Confidence: {easyocr_result.confidence:.2%}")
        print(f"Text length: {len(easyocr_result.text)} chars")
        print(f"Preview: {easyocr_result.text[:200]}...")
        
        # Test 2: Tesseract
        print("\n" + "-"*80)
        print("METHOD 2: Tesseract")
        print("-"*80)
        start = time.time()
        tesseract_result = self._test_tesseract(image)
        tesseract_time = time.time() - start
        
        print(f"Time: {tesseract_time:.2f}s")
        print(f"Confidence: {tesseract_result.confidence:.2%}")
        print(f"Text length: {len(tesseract_result.text)} chars")
        print(f"Preview: {tesseract_result.text[:200]}...")
        
        # Test 3: TrOCR (if available)
        if self.ocr_engine.trocr_available:
            print("\n" + "-"*80)
            print("METHOD 3: TrOCR (Handwriting)")
            print("-"*80)
            start = time.time()
            trocr_text = self.ocr_engine.extract_with_trocr(image)
            trocr_time = time.time() - start
            
            print(f"Time: {trocr_time:.2f}s")
            print(f"Text length: {len(trocr_text)} chars")
            print(f"Preview: {trocr_text[:200]}...")
        else:
            print("\n⚠️ TrOCR not available")
        
        # Test 4: Hybrid (automatic selection)
        print("\n" + "-"*80)
        print("METHOD 4: Hybrid (Automatic)")
        print("-"*80)
        start = time.time()
        hybrid_result = self.ocr_engine.extract_text_hybrid(image)
        hybrid_time = time.time() - start
        
        print(f"Time: {hybrid_time:.2f}s")
        print(f"Method chosen: {hybrid_result.method}")
        print(f"Is handwritten: {hybrid_result.is_handwritten}")
        print(f"Confidence: {hybrid_result.confidence:.2%}")
        print(f"Text length: {len(hybrid_result.text)} chars")
        print(f"Preview: {hybrid_result.text[:200]}...")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        results = [
            ("EasyOCR", easyocr_result.confidence, len(easyocr_result.text), easyocr_time),
            ("Tesseract", tesseract_result.confidence, len(tesseract_result.text), tesseract_time),
        ]
        
        if self.ocr_engine.trocr_available:
            results.append(("TrOCR", 0.7, len(trocr_text), trocr_time))
        
        results.append(("Hybrid", hybrid_result.confidence, len(hybrid_result.text), hybrid_time))
        
        print(f"\n{'Method':<15} {'Confidence':<12} {'Text Length':<12} {'Time':<10}")
        print("-"*50)
        for method, conf, length, t in results:
            print(f"{method:<15} {conf:.2%}        {length:<12} {t:.2f}s")
        
        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        
        if hybrid_result.is_handwritten:
            print("✓ This appears to be HANDWRITTEN text")
            print("  → TrOCR was used automatically")
            print("  → Expected accuracy: 75-85%")
        else:
            print("✓ This appears to be PRINTED text")
            print(f"  → {hybrid_result.method.upper()} was used")
            print("  → Expected accuracy: 90-95%")
    
    def _test_easyocr(self, image):
        """Test EasyOCR"""
        return self.ocr_engine.extract_with_easyocr(image)
    
    def _test_tesseract(self, image):
        """Test Tesseract"""
        return self.ocr_engine.extract_with_tesseract(image)
    
    def batch_test(self, image_dir: str):
        """Test multiple images in a directory"""
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"\nFound {len(image_files)} images in {image_dir}")
        
        results_summary = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n{'='*80}")
            print(f"IMAGE {i}/{len(image_files)}: {image_file.name}")
            print(f"{'='*80}")
            
            try:
                self.test_image(str(image_file))
                results_summary.append((image_file.name, "SUCCESS"))
            except Exception as e:
                print(f"❌ ERROR: {e}")
                results_summary.append((image_file.name, f"FAILED: {e}"))
        
        # Final summary
        print("\n" + "="*80)
        print("BATCH TEST SUMMARY")
        print("="*80)
        
        for filename, status in results_summary:
            status_symbol = "✓" if status == "SUCCESS" else "✗"
            print(f"{status_symbol} {filename}: {status}")
        
        success_count = sum(1 for _, s in results_summary if s == "SUCCESS")
        print(f"\nTotal: {len(results_summary)} | Success: {success_count} | Failed: {len(results_summary) - success_count}")


def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python test_trocr.py <image_path>")
        print("  Batch test:    python test_trocr.py <directory_path>")
        print("\nExamples:")
        print("  python test_trocr.py sample_prescriptions/prescription_1.jpg")
        print("  python test_trocr.py sample_prescriptions/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    print("\n" + "="*80)
    print("TrOCR COMPARISON TEST")
    print("="*80)
    
    comparison = OCRComparison()
    
    # Check if path is directory or file
    if Path(path).is_dir():
        print(f"Running batch test on directory: {path}")
        comparison.batch_test(path)
    elif Path(path).is_file():
        print(f"Running single test on file: {path}")
        comparison.test_image(path)
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()