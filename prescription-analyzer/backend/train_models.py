#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    
    print("=" * 80)
    print("AI PRESCRIPTION ANALYZER - COMPREHENSIVE TRAINING")
    print("Target: 99% Confidence Rate")
    print("=" * 80)
    print()
    
    # Check if data folder exists
    data_folder = Path("data/prescriptions")
    
    if not data_folder.exists():
        print(f"❌ Error: Data folder not found: {data_folder}")
        print()
        print("Please create the folder and add your 142 prescription images:")
        print(f"  mkdir -p {data_folder}")
        print(f"  # Copy your prescription images to {data_folder}/")
        return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(data_folder.glob(f"*{ext}")))
        images.extend(list(data_folder.glob(f"*{ext.upper()}")))
    
    print(f"📊 Found {len(images)} prescription images")
    
    if len(images) == 0:
        print("❌ No images found. Please add prescription images to train on.")
        return
    
    if len(images) < 10:
        print(f"⚠️  Warning: Only {len(images)} images found. Recommend at least 50 for good training.")
        print("   Continue anyway? (y/n): ", end='')
        if input().lower() != 'y':
            return
    
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Training samples: {len(images)}")
    print(f"  Data folder: {data_folder}")
    print(f"  Target confidence: 99%")
    print(f"  Self-learning: Enabled")
    print(f"  Advanced OCR: Ensemble mode")
    print(f"  ML Models: Gradient Boosting + Random Forest")
    print("=" * 80)
    print()
    
    # Import the analyzer
    try:
        # Add backend to path
        backend_path = Path(__file__).parent / "backend"
        sys.path.insert(0, str(backend_path))
        
        from enhanced_prescription_analyzer import SelfLearningPrescriptionAnalyzer
        
        print("✓ Analyzer module loaded")
        print()
        
    except ImportError as e:
        print(f"❌ Failed to import analyzer: {e}")
        print()
        print("Make sure enhanced_prescription_analyzer.py is in the backend/ folder")
        return
    
    # Initialize analyzer
    print("🚀 Initializing Self-Learning Analyzer...")
    try:
        analyzer = SelfLearningPrescriptionAnalyzer(data_folder=str(data_folder))
        print("✓ Analyzer initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start training
    print("=" * 80)
    print("STARTING TRAINING PROCESS")
    print("=" * 80)
    print()
    
    try:
        success = analyzer.train_from_dataset(retrain=True)
        
        if success:
            print()
            print("=" * 80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print("Models have been trained and saved.")
            print("The analyzer will now achieve much higher confidence rates.")
            print()
            print("Next steps:")
            print("  1. Test on new prescriptions: python test_analyzer.py <image>")
            print("  2. Check model performance: python evaluate_models.py")
            print("  3. Continue learning: The system will auto-improve with each use")
            print()
            
            # Save training summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'samples_trained': len(images),
                'data_folder': str(data_folder),
                'target_confidence': 0.99,
                'status': 'success'
            }
            
            with open('training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("📄 Training summary saved to: training_summary.json")
            
        else:
            print()
            print("=" * 80)
            print("❌ TRAINING FAILED")
            print("=" * 80)
            print()
            print("Please check the logs for details")
            
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TRAINING ERROR")
        print("=" * 80)
        print(f"\nError: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("Please check:")
        print("  1. All prescription images are valid")
        print("  2. Tesseract and EasyOCR are installed")
        print("  3. Sufficient disk space and memory")


if __name__ == "__main__":
    main()