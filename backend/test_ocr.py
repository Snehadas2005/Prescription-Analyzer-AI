#!/usr/bin/env python3
"""
Quick OCR Diagnostic Test
Run this to check if OCR is working properly
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("OCR DIAGNOSTIC TEST")
print("=" * 80)
print()

# Test 1: Check Python packages
print("Step 1: Checking installed packages...")
packages_to_check = [
    'cv2',
    'numpy',
    'PIL',
    'easyocr',
    'pytesseract',
]

missing_packages = []
for package in packages_to_check:
    try:
        __import__(package)
        print(f"  ✓ {package} installed")
    except ImportError:
        print(f"  ✗ {package} MISSING")
        missing_packages.append(package)

if missing_packages:
    print()
    print("❌ MISSING PACKAGES:", ', '.join(missing_packages))
    print("Install with: pip install opencv-python numpy Pillow easyocr pytesseract")
    sys.exit(1)

print()

# Test 2: Check if prescription_analyzer can be imported
print("Step 2: Checking prescription_analyzer...")
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
    print("  ✓ EnhancedPrescriptionAnalyzer imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

print()

# Test 3: Initialize analyzer
print("Step 3: Initializing analyzer...")
try:
    analyzer = EnhancedPrescriptionAnalyzer(force_api=False)
    print("  ✓ Analyzer initialized")
except Exception as e:
    print(f"  ✗ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test 4: Check EasyOCR
print("Step 4: Testing EasyOCR...")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("  ✓ EasyOCR reader created")
    
    # Test on simple image
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Create test image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 30), "TEST 123", fill='black', font=font)
    
    # Convert to numpy
    img_array = np.array(img)
    
    # Test OCR
    results = reader.readtext(img_array, detail=0)
    
    if results:
        print(f"  ✓ EasyOCR working! Detected: {results}")
    else:
        print("  ⚠️ EasyOCR initialized but detected no text from test image")
        
except Exception as e:
    print(f"  ✗ EasyOCR test failed: {e}")
    print("  Try reinstalling: pip install easyocr")

print()

# Test 5: Check Tesseract
print("Step 5: Testing Tesseract...")
try:
    import pytesseract
    from PIL import Image, ImageDraw
    
    # Create test image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TEST 123", fill='black')
    
    # Test OCR
    text = pytesseract.image_to_string(img)
    
    if text.strip():
        print(f"  ✓ Tesseract working! Detected: '{text.strip()}'")
    else:
        print("  ⚠️ Tesseract initialized but detected no text")
        
except Exception as e:
    print(f"  ✗ Tesseract test failed: {e}")
    print("  Install Tesseract:")
    print("    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("    Linux: sudo apt-get install tesseract-ocr")
    print("    Mac: brew install tesseract")

print()

# Test 6: Test on actual prescription if provided
print("Step 6: Testing on real prescription image...")
if len(sys.argv) > 1:
    test_image = sys.argv[1]
    if os.path.exists(test_image):
        print(f"  Testing on: {test_image}")
        
        try:
            result = analyzer.analyze_prescription(test_image)
            
            print(f"  Success: {result.success}")
            print(f"  Confidence: {result.confidence_score:.2%}")
            print(f"  Patient: {result.patient.name}")
            print(f"  Doctor: {result.doctor.name}")
            print(f"  Medicines: {len(result.medicines)}")
            print(f"  Raw text length: {len(result.raw_text)} chars")
            
            if result.raw_text:
                print(f"  Raw text preview: {result.raw_text[:200]}...")
            else:
                print("  ⚠️ NO TEXT EXTRACTED!")
                
        except Exception as e:
            print(f"  ✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ File not found: {test_image}")
else:
    print("  Skipped - no image provided")
    print("  Usage: python test_ocr.py path/to/prescription.jpg")

print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print()

if missing_packages:
    print("⚠️ Fix missing packages first!")
else:
    print("✅ All basic checks passed!")
    print()
    print("If you still get 0% confidence, the issue might be:")
    print("  1. Image quality too low")
    print("  2. OCR not recognizing the text format")
    print("  3. Prescription is handwritten (needs TrOCR)")