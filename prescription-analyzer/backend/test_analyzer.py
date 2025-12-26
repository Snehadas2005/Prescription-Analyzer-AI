#!/usr/bin/env python3
import sys
from pathlib import Path
from prescription_analyzer import EnhancedPrescriptionAnalyzer

def test_prescription(image_path: str):
    print(f"\n{'='*80}")
    print(f"TESTING: {image_path}")
    print(f"{'='*80}\n")
    
    analyzer = EnhancedPrescriptionAnalyzer()
    result = analyzer.analyze_prescription(image_path)
    
    if result.success:
        print(f"✅ SUCCESS - Confidence: {result.confidence_score:.2%}")
        print(f"\nBreakdown:")
        print(f"  - OCR Confidence:        {result.ocr_confidence:.2%}")
        print(f"  - Extraction Confidence: {result.extraction_confidence:.2%}")
        print(f"  - Validation Confidence: {result.validation_confidence:.2%}")
        print(f"\n📊 Extracted Data:")
        print(f"\n  Patient: {result.patient.name} (Age: {result.patient.age}, Gender: {result.patient.gender})")
        print(f"           Confidence: {result.patient.confidence:.2%}")
        print(f"\n  Doctor:  {result.doctor.name} ({result.doctor.specialization})")
        print(f"           Confidence: {result.doctor.confidence:.2%}")
        print(f"\n  Medicines: {len(result.medicines)} found")
        for i, med in enumerate(result.medicines, 1):
            print(f"    {i}. {med.name} - {med.dosage} - {med.frequency}")
            print(f"       Confidence: {med.confidence:.2%}")
    else:
        print(f"❌ FAILED: {result.error}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_analyzer.py <prescription_image_path>")
        sys.exit(1)
    
    test_prescription(sys.argv[1])