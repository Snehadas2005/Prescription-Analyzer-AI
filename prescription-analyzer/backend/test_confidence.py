#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import json
from datetime import datetime

def test_prescription(image_path: str):
    """Test a single prescription"""
    
    print("=" * 80)
    print(f"TESTING: {Path(image_path).name}")
    print("=" * 80)
    print()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    # Import analyzer
    try:
        backend_path = Path(__file__).parent / "backend"
        sys.path.insert(0, str(backend_path))
        from enhanced_prescription_analyzer import SelfLearningPrescriptionAnalyzer
        
        analyzer = SelfLearningPrescriptionAnalyzer()
        print("✓ Analyzer loaded with trained models")
        print()
        
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return None
    
    # Analyze
    print("🔍 Analyzing prescription...")
    print()
    
    try:
        result = analyzer.analyze_prescription(image_path)
        
        # Display results
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        print(f"📊 Overall Confidence: {result.confidence_score:.1%}")
        print(f"   - OCR Confidence: {result.ocr_confidence:.1%}")
        print(f"   - Extraction Confidence: {result.extraction_confidence:.1%}")
        print(f"   - Validation Confidence: {result.validation_confidence:.1%}")
        print()
        
        # Confidence breakdown
        if result.confidence_score >= 0.95:
            status = "✅ EXCELLENT"
            color = "green"
        elif result.confidence_score >= 0.85:
            status = "✓ GOOD"
            color = "yellow"
        elif result.confidence_score >= 0.70:
            status = "⚠ ACCEPTABLE"
            color = "yellow"
        else:
            status = "❌ LOW"
            color = "red"
        
        print(f"Status: {status} ({result.confidence_score:.1%})")
        print()
        
        # Patient info
        print("👤 PATIENT INFORMATION")
        print(f"   Name:   {result.patient.name or 'Not detected'} (confidence: {result.patient.confidence:.0%})")
        print(f"   Age:    {result.patient.age or 'Not detected'}")
        print(f"   Gender: {result.patient.gender or 'Not detected'}")
        print()
        
        # Doctor info
        print("👨‍⚕️ DOCTOR INFORMATION")
        print(f"   Name:           {result.doctor.name or 'Not detected'} (confidence: {result.doctor.confidence:.0%})")
        print(f"   Specialization: {result.doctor.specialization or 'Not detected'}")
        print(f"   Registration:   {result.doctor.registration_number or 'Not detected'}")
        print()
        
        # Medicines
        print(f"💊 MEDICINES ({len(result.medicines)} detected)")
        if result.medicines:
            for i, med in enumerate(result.medicines, 1):
                print(f"\n   {i}. {med.name} (confidence: {med.confidence:.0%})")
                print(f"      Dosage:    {med.dosage}")
                print(f"      Frequency: {med.frequency}")
                print(f"      Duration:  {med.duration}")
                if med.instructions:
                    print(f"      Instructions: {med.instructions}")
        else:
            print("   No medicines detected")
        
        print()
        print("=" * 80)
        
        # Improvement suggestions
        if result.confidence_score < 0.95:
            print("\n📋 IMPROVEMENT SUGGESTIONS:")
            if result.ocr_confidence < 0.80:
                print("  • Image quality could be improved (better lighting, focus)")
            if result.extraction_confidence < 0.85:
                print("  • Some information may be handwritten (harder to extract)")
            if result.validation_confidence < 0.90:
                print("  • Some extracted values couldn't be validated against database")
            print()
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_test(test_folder: str):
    """Test multiple prescriptions"""
    
    test_path = Path(test_folder)
    if not test_path.exists():
        print(f"❌ Folder not found: {test_folder}")
        return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(test_path.glob(f"*{ext}")))
        images.extend(list(test_path.glob(f"*{ext.upper()}")))
    
    if not images:
        print(f"❌ No images found in {test_folder}")
        return
    
    print("=" * 80)
    print(f"BATCH TESTING: {len(images)} prescriptions")
    print("=" * 80)
    print()
    
    results = []
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Testing {img_path.name}...")
        result = test_prescription(str(img_path))
        
        if result:
            results.append({
                'file': img_path.name,
                'confidence': result.confidence_score,
                'patient': result.patient.name,
                'doctor': result.doctor.name,
                'medicines_count': len(result.medicines)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("BATCH TEST SUMMARY")
    print("=" * 80)
    print()
    
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        high_conf = sum(1 for r in results if r['confidence'] >= 0.95)
        good_conf = sum(1 for r in results if 0.85 <= r['confidence'] < 0.95)
        low_conf = sum(1 for r in results if r['confidence'] < 0.85)
        
        print(f"Total tested: {len(results)}")
        print(f"Average confidence: {avg_conf:.1%}")
        print(f"  ✅ Excellent (≥95%): {high_conf}")
        print(f"  ✓  Good (85-95%): {good_conf}")
        print(f"  ⚠  Low (<85%): {low_conf}")
        print()
        
        # Save results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total': len(results),
                'average_confidence': avg_conf,
                'results': results
            }, f, indent=2)
        
        print(f"📄 Results saved to: {results_file}")
        print()


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single test:  python test_confidence.py <image_path>")
        print("  Batch test:   python test_confidence.py <folder_path>")
        print()
        print("Examples:")
        print("  python test_confidence.py test_prescriptions/rx1.jpg")
        print("  python test_confidence.py test_prescriptions/")
        return
    
    path = sys.argv[1]
    
    if Path(path).is_dir():
        batch_test(path)
    elif Path(path).is_file():
        test_prescription(path)
    else:
        print(f"❌ Path not found: {path}")


if __name__ == "__main__":
    main()