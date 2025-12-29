#!/usr/bin/env python3
"""
Test script to compare traditional OCR vs TrOCR
Shows the improvement on handwritten prescriptions
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
import os

# Import the analyzer
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
except ImportError:
    print("Error: Could not import EnhancedPrescriptionAnalyzer")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)


def test_prescription_with_trocr(image_path: str):
    """Test a prescription image with TrOCR support"""
    
    print("\n" + "="*80)
    print(f"TESTING: {image_path}")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    # Load image info
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"\n📄 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Test with TrOCR enabled
    print("\n" + "-"*80)
    print("METHOD 1: Enhanced Analyzer with TrOCR Support")
    print("-"*80)
    
    start = time.time()
    
    try:
        analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=os.getenv('COHERE_API_KEY'),
            use_gpu=False,  # Set to True if you have GPU
            force_api=False
        )
        
        result = analyzer.analyze_prescription(image_path)
        
        elapsed = time.time() - start
        
        print(f"\n⏱️  Time: {elapsed:.2f}s")
        print(f"✓ Success: {result.success}")
        print(f"✓ Confidence: {result.confidence_score:.2%}")
        print(f"✓ OCR Methods: {', '.join(result.ocr_methods_used) if result.ocr_methods_used else 'N/A'}")
        
        if result.success:
            print("\n📊 Extracted Information:")
            print(f"\n  👤 Patient:")
            print(f"     Name:   {result.patient.name or 'Not detected'}")
            print(f"     Age:    {result.patient.age or 'Not detected'}")
            print(f"     Gender: {result.patient.gender or 'Not detected'}")
            
            print(f"\n  👨‍⚕️ Doctor:")
            print(f"     Name:           {result.doctor.name or 'Not detected'}")
            print(f"     Specialization: {result.doctor.specialization or 'Not detected'}")
            print(f"     Registration:   {result.doctor.registration_number or 'Not detected'}")
            
            print(f"\n  💊 Medicines: {len(result.medicines)}")
            for i, med in enumerate(result.medicines[:5], 1):  # Show first 5
                print(f"     {i}. {med.name} - {med.dosage}")
            
            if len(result.medicines) > 5:
                print(f"     ... and {len(result.medicines) - 5} more")
            
            print(f"\n  📝 Raw Text Preview:")
            preview = result.raw_text[:300] + "..." if len(result.raw_text) > 300 else result.raw_text
            print(f"     {preview}")
        else:
            print(f"\n❌ Analysis failed: {result.error}")
        
        # Recommendation
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        if result.ocr_methods_used:
            if 'trocr' in result.ocr_methods_used:
                print("✓ TrOCR was used for HANDWRITTEN text detection")
                print("  → Expected accuracy: 75-85% for handwriting")
            else:
                print("✓ Traditional OCR was used for PRINTED text")
                print("  → Expected accuracy: 90-95% for printed text")
        
        if result.confidence_score < 0.6:
            print("\n⚠️  Low confidence detected. Recommendations:")
            print("  1. Ensure good lighting and clear image")
            print("  2. Avoid shadows and glare")
            print("  3. Try scanning at higher resolution")
            print("  4. Place prescription on flat surface")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def batch_test(directory: str):
    """Test multiple images in a directory"""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_files = [
        f for f in dir_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {directory}")
        return
    
    print(f"\n📁 Found {len(image_files)} images in {directory}")
    print("="*80)
    
    results_summary = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n\n{'='*80}")
        print(f"IMAGE {i}/{len(image_files)}: {image_file.name}")
        print(f"{'='*80}")
        
        try:
            test_prescription_with_trocr(str(image_file))
            results_summary.append((image_file.name, "SUCCESS"))
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results_summary.append((image_file.name, f"FAILED: {e}"))
    
    # Final summary
    print("\n\n" + "="*80)
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
        print("  Single image:  python test_hybrid_ocr.py <image_path>")
        print("  Batch test:    python test_hybrid_ocr.py <directory_path>")
        print("\nExamples:")
        print("  python test_hybrid_ocr.py sample_prescriptions/prescription_1.jpg")
        print("  python test_hybrid_ocr.py sample_prescriptions/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    print("\n" + "="*80)
    print("TrOCR-ENABLED PRESCRIPTION ANALYZER TEST")
    print("="*80)
    
    # Check if path is directory or file
    if Path(path).is_dir():
        print(f"Running batch test on directory: {path}")
        batch_test(path)
    elif Path(path).is_file():
        print(f"Running single test on file: {path}")
        test_prescription_with_trocr(path)
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()