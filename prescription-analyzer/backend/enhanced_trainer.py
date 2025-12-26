#!/usr/bin/env python3
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import pickle
from collections import defaultdict
import cv2

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prescription_analyzer import EnhancedPrescriptionAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfLearningTrainer:

    def __init__(self, data_folder: str = "../data/prescriptions"):
        self.data_folder = Path(data_folder)
        self.models_folder = Path("models/trained")
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        # Knowledge bases
        self.medicine_patterns = defaultdict(int)
        self.doctor_patterns = defaultdict(int)
        self.patient_patterns = defaultdict(int)
        self.dosage_patterns = defaultdict(int)
        self.frequency_patterns = defaultdict(int)
        
        # Training history
        self.training_history = []
        
    def train_from_prescriptions(self):
        """
        Main training method - learns from all prescriptions
        """
        logger.info("="*80)
        logger.info("ENHANCED SELF-LEARNING TRAINING")
        logger.info("="*80)
        
        # Find all prescription images
        image_files = self._find_prescription_images()
        
        if len(image_files) < 10:
            logger.error(f"Not enough training data! Found only {len(image_files)} images")
            logger.error(f"Expected at least 142 prescriptions in {self.data_folder}")
            return False
        
        logger.info(f"Found {len(image_files)} prescription images")
        logger.info("Starting multi-pass training...")
        
        # Initialize analyzer
        analyzer = EnhancedPrescriptionAnalyzer(force_api=False)
        
        # Pass 1: Extract all data and build knowledge bases
        logger.info("\n" + "="*80)
        logger.info("PASS 1: Building Knowledge Bases")
        logger.info("="*80)
        
        training_data = []
        for i, img_path in enumerate(image_files, 1):
            logger.info(f"\nProcessing {i}/{len(image_files)}: {img_path.name}")
            
            try:
                result = analyzer.analyze_prescription(str(img_path))
                
                if result.success:
                    # Extract patterns
                    self._extract_patterns(result)
                    
                    # Store for later passes
                    training_data.append({
                        'path': str(img_path),
                        'result': result,
                        'raw_text': result.raw_text
                    })
                    
                    logger.info(f"  ✓ Patient: {result.patient.name or 'N/A'}")
                    logger.info(f"  ✓ Doctor: {result.doctor.name or 'N/A'}")
                    logger.info(f"  ✓ Medicines: {len(result.medicines)}")
                else:
                    logger.warning(f"  ✗ Analysis failed: {result.error}")
                    
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
        
        logger.info(f"\n✓ Pass 1 Complete: Processed {len(training_data)} prescriptions")
        
        # Save knowledge bases
        self._save_knowledge_bases()
        
        # Pass 2: Learn from patterns and improve
        logger.info("\n" + "="*80)
        logger.info("PASS 2: Pattern Learning and Optimization")
        logger.info("="*80)
        
        self._learn_from_patterns()
        
        # Pass 3: Validate and calculate confidence
        logger.info("\n" + "="*80)
        logger.info("PASS 3: Validation and Confidence Calculation")
        logger.info("="*80)
        
        validation_results = self._validate_training(training_data)
        
        # Generate training report
        self._generate_training_report(validation_results)
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        
        return True
    
    def _find_prescription_images(self) -> List[Path]:
        """Find all prescription images"""
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
        images = []
        
        for ext in extensions:
            images.extend(self.data_folder.glob(f"*{ext}"))
            images.extend(self.data_folder.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _extract_patterns(self, result):
        """Extract and count patterns from analysis result"""
        # Medicine patterns
        for med in result.medicines:
            if med.name:
                self.medicine_patterns[med.name.lower()] += 1
            if med.dosage:
                self.dosage_patterns[med.dosage.lower()] += 1
            if med.frequency:
                self.frequency_patterns[med.frequency.lower()] += 1
        
        # Doctor patterns
        if result.doctor.name:
            self.doctor_patterns[result.doctor.name.lower()] += 1
        if result.doctor.specialization:
            self.doctor_patterns[f"spec:{result.doctor.specialization.lower()}"] += 1
        
        # Patient patterns (gender, age ranges)
        if result.patient.gender:
            self.patient_patterns[f"gender:{result.patient.gender.lower()}"] += 1
        if result.patient.age:
            try:
                age = int(result.patient.age)
                age_range = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                self.patient_patterns[f"age_range:{age_range}"] += 1
            except:
                pass
    
    def _save_knowledge_bases(self):
        """Save learned knowledge bases"""
        knowledge = {
            'medicines': dict(self.medicine_patterns),
            'dosages': dict(self.dosage_patterns),
            'frequencies': dict(self.frequency_patterns),
            'doctors': dict(self.doctor_patterns),
            'patients': dict(self.patient_patterns),
            'timestamp': datetime.now().isoformat(),
            'total_patterns': sum(len(p) for p in [
                self.medicine_patterns,
                self.dosage_patterns,
                self.frequency_patterns,
                self.doctor_patterns,
                self.patient_patterns
            ])
        }
        
        # Save as pickle for fast loading
        with open(self.models_folder / 'knowledge_base.pkl', 'wb') as f:
            pickle.dump(knowledge, f)
        
        # Save as JSON for readability
        with open(self.models_folder / 'knowledge_base.json', 'w') as f:
            json.dump(knowledge, f, indent=2)
        
        logger.info(f"✓ Saved knowledge base with {knowledge['total_patterns']} patterns")
        logger.info(f"  - Medicines: {len(self.medicine_patterns)}")
        logger.info(f"  - Dosages: {len(self.dosage_patterns)}")
        logger.info(f"  - Frequencies: {len(self.frequency_patterns)}")
    
    def _learn_from_patterns(self):
        """Learn improved extraction rules from patterns"""
        # Top medicines (most common)
        top_medicines = sorted(
            self.medicine_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:100]
        
        # Top dosage patterns
        top_dosages = sorted(
            self.dosage_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]
        
        # Top frequency patterns
        top_frequencies = sorted(
            self.frequency_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Save learned patterns for quick lookup
        learned_patterns = {
            'top_medicines': [m[0] for m in top_medicines],
            'top_dosages': [d[0] for d in top_dosages],
            'top_frequencies': [f[0] for f in top_frequencies],
            'medicine_frequency_map': {
                m[0]: m[1] for m in top_medicines
            }
        }
        
        with open(self.models_folder / 'learned_patterns.json', 'w') as f:
            json.dump(learned_patterns, f, indent=2)
        
        logger.info("✓ Learned and saved extraction patterns")
    
    def _validate_training(self, training_data: List[Dict]) -> Dict:
        """Validate training results"""
        results = {
            'total': len(training_data),
            'with_patient': 0,
            'with_doctor': 0,
            'with_medicines': 0,
            'full_extraction': 0,
            'avg_confidence': 0.0,
            'medicine_extraction_rate': 0.0
        }
        
        confidences = []
        medicine_counts = []
        
        for data in training_data:
            result = data['result']
            
            has_patient = bool(result.patient.name or result.patient.age)
            has_doctor = bool(result.doctor.name)
            has_medicines = len(result.medicines) > 0
            
            if has_patient:
                results['with_patient'] += 1
            if has_doctor:
                results['with_doctor'] += 1
            if has_medicines:
                results['with_medicines'] += 1
                medicine_counts.append(len(result.medicines))
            if has_patient and has_doctor and has_medicines:
                results['full_extraction'] += 1
            
            confidences.append(result.confidence_score)
        
        results['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        results['medicine_extraction_rate'] = (
            np.mean(medicine_counts) if medicine_counts else 0.0
        )
        
        # Calculate percentages
        results['patient_extraction_rate'] = (results['with_patient'] / results['total']) * 100
        results['doctor_extraction_rate'] = (results['with_doctor'] / results['total']) * 100
        results['medicine_extraction_rate_pct'] = (results['with_medicines'] / results['total']) * 100
        results['full_extraction_rate'] = (results['full_extraction'] / results['total']) * 100
        
        return results
    
    def _generate_training_report(self, results: Dict):
        """Generate comprehensive training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'training_results': results,
            'knowledge_base_size': {
                'medicines': len(self.medicine_patterns),
                'dosages': len(self.dosage_patterns),
                'frequencies': len(self.frequency_patterns),
                'doctors': len(self.doctor_patterns)
            },
            'top_medicines': sorted(
                self.medicine_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'model_version': '2.0.0',
            'target_confidence': 0.99,
            'achieved_confidence': results['avg_confidence']
        }
        
        # Save report
        report_path = self.models_folder / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Prescriptions: {results['total']}")
        logger.info(f"Patient Extraction: {results['patient_extraction_rate']:.1f}%")
        logger.info(f"Doctor Extraction: {results['doctor_extraction_rate']:.1f}%")
        logger.info(f"Medicine Extraction: {results['medicine_extraction_rate_pct']:.1f}%")
        logger.info(f"Full Extraction: {results['full_extraction_rate']:.1f}%")
        logger.info(f"Average Confidence: {results['avg_confidence']:.2%}")
        logger.info(f"Avg Medicines/Prescription: {results['medicine_extraction_rate']:.1f}")
        logger.info(f"\nKnowledge Base:")
        logger.info(f"  - Unique Medicines: {len(self.medicine_patterns)}")
        logger.info(f"  - Dosage Patterns: {len(self.dosage_patterns)}")
        logger.info(f"  - Frequency Patterns: {len(self.frequency_patterns)}")
        logger.info(f"\nReport saved: {report_path}")


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("ENHANCED SELF-LEARNING PRESCRIPTION ANALYZER TRAINING")
    print("="*80)
    print()
    
    # Check for data folder
    data_folder = Path("../data/prescriptions")
    if not data_folder.exists():
        print(f"❌ Data folder not found: {data_folder}")
        print(f"Please create the folder and add prescription images")
        return
    
    # Initialize and run trainer
    trainer = SelfLearningTrainer(data_folder=str(data_folder))
    
    success = trainer.train_from_prescriptions()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("The model has learned from your prescriptions.")
        print("Run analysis again to see improved confidence scores.")
    else:
        print("\n❌ Training failed. Check logs for details.")


if __name__ == "__main__":
    main()