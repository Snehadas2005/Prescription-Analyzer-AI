#!/usr/bin/env python3
"""Test TrOCR integration"""

from hybrid_ocr import HybridOCREngine
import cv2
import sys

def test_hybrid_ocr(image_path: str):
    print(f"Testing hybrid OCR on: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Failed to load {image_path}")
        return
    
    # Initialize engine
    engine = HybridOCREngine(use_gpu=False)
    
    # Extract
    text, confidence, methods = engine.extract_hybrid(image)
    
    # Results
    print(f"\nMethods used: {methods}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Extracted text: {text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_trocr.py <image_path>")
        sys.exit(1)
    
    test_hybrid_ocr(sys.argv[1])