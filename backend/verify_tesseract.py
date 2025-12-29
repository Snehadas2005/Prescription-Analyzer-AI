#!/usr/bin/env python3
"""
Quick Tesseract Verification Script
Run this to verify Tesseract is working
"""

import sys
import os

print("=" * 80)
print("TESSERACT VERIFICATION TEST")
print("=" * 80)
print()

# Step 1: Check if pytesseract is installed
print("Step 1: Checking pytesseract package...")
try:
    import pytesseract
    print("  ✓ pytesseract package installed")
except ImportError:
    print("  ✗ pytesseract not installed")
    print("  Install: pip install pytesseract")
    sys.exit(1)

print()

# Step 2: Find Tesseract executable
print("Step 2: Looking for Tesseract executable...")

if os.name == 'nt':  # Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    
    tesseract_path = None
    for path in possible_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            tesseract_path = path
            print(f"  ✓ Found at: {path}")
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
    if not tesseract_path:
        print("  ✗ Tesseract not found in common paths!")
        print()
        print("  Please install Tesseract:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        print()
        print("  After installation, add to PATH or set manually:")
        print("  pytesseract.pytesseract.tesseract_cmd = r'C:\\Path\\To\\tesseract.exe'")
        sys.exit(1)
else:
    # Linux/Mac - should be in PATH
    import subprocess
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        print(f"  ✓ Tesseract found in PATH")
        print(f"  Version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
    except FileNotFoundError:
        print("  ✗ Tesseract not found in PATH")
        print("  Install: sudo apt-get install tesseract-ocr")
        sys.exit(1)

print()

# Step 3: Test Tesseract with a simple image
print("Step 3: Testing Tesseract OCR...")

try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple test image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 30), "TEST 123 ABC", fill='black', font=font)
    
    # Run OCR
    text = pytesseract.image_to_string(img)
    
    print(f"  Test image text: 'TEST 123 ABC'")
    print(f"  OCR detected: '{text.strip()}'")
    
    if text.strip():
        print("  ✓ Tesseract is working!")
    else:
        print("  ⚠️ Tesseract didn't detect any text")
        print("  This might be an issue with Tesseract installation")
        
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    sys.exit(1)

print()

# Step 4: Test on actual prescription if provided
print("Step 4: Testing on real prescription (if provided)...")

if len(sys.argv) > 1:
    prescription_path = sys.argv[1]
    
    if os.path.exists(prescription_path):
        print(f"  Loading: {prescription_path}")
        
        try:
            import cv2
            
            # Read image
            img = cv2.imread(prescription_path)
            
            if img is None:
                print("  ✗ Failed to read image")
                sys.exit(1)
            
            print(f"  Image size: {img.shape}")
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Run OCR
            text = pytesseract.image_to_string(gray)
            
            print(f"  Extracted text length: {len(text)} characters")
            
            if text.strip():
                print(f"  Text preview: {text[:200]}...")
                print("  ✓ Successfully extracted text from prescription!")
            else:
                print("  ⚠️ No text extracted from prescription")
                print("  This could mean:")
                print("    - Image quality is too low")
                print("    - Image is handwritten (needs different OCR)")
                print("    - Image preprocessing needed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ✗ File not found: {prescription_path}")
else:
    print("  Skipped - no prescription image provided")
    print("  Usage: python verify_tesseract.py path/to/prescription.jpg")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()

if tesseract_path:
    print("✅ Tesseract is installed and working!")
    print()
    print("Next steps:")
    print("  1. Replace backend/prescription_analyzer.py with the fixed version")
    print("  2. Restart ML service: python -m uvicorn app.main:app --reload --port 8000")
    print("  3. Upload a prescription and check the logs")
else:
    print("⚠️ Tesseract verification incomplete")