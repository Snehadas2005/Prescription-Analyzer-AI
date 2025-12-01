import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from pymongo import MongoClient
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.data_path = Path('ml-service/data')
        self.training_path = self.data_path / 'training'
        self.feedback_path = self.data_path / 'feedback'
        self.raw_path = self.data_path / 'raw'
        
        # Create directories
        self.training_path.mkdir(parents=True, exist_ok=True)
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        # MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
        self.db = self.mongo_client['prescription_analyzer']
    
    def build_training_dataset(self) -> Tuple[pd.DataFrame, int]:
        """Build training dataset from verified feedback"""
        logger.info("Building training dataset from feedback...")
        
        # Get verified feedback from database
        feedback_collection = self.db['feedback']
        verified_feedback = list(feedback_collection.find({
            'feedback_type': {'$in': ['correction', 'confirmation']}
        }))
        
        logger.info(f"Found {len(verified_feedback)} verified feedback entries")
        
        training_samples = []
        
        for feedback in verified_feedback:
            try:
                sample = self._process_feedback_to_training_sample(feedback)
                if sample:
                    training_samples.append(sample)
            except Exception as e:
                logger.error(f"Error processing feedback {feedback.get('_id')}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(training_samples)
        
        if len(df) > 0:
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = self.training_path / f'training_dataset_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved training dataset to {csv_path}")
            
            # Also update the latest dataset
            latest_path = self.training_path / 'latest_dataset.csv'
            df.to_csv(latest_path, index=False)
        
        return df, len(training_samples)
    
    def _process_feedback_to_training_sample(self, feedback: Dict) -> Dict[str, Any]:
        """Process a single feedback entry into training sample"""
        prescription_id = feedback['prescription_id']
        
        # Get original prescription
        prescription = self.db['prescriptions'].find_one({
            'prescription_id': prescription_id
        })
        
        if not prescription:
            return None
        
        # Get the corrected or confirmed data
        if feedback['feedback_type'] == 'correction' and feedback.get('corrections'):
            # Use corrected data as ground truth
            ground_truth = feedback['corrections']
        elif feedback['feedback_type'] == 'confirmation':
            # Use original extraction as ground truth (confirmed correct)
            ground_truth = {
                'patient': prescription['patient'],
                'doctor': prescription['doctor'],
                'medicines': prescription['medicines']
            }
        else:
            return None
        
        # Create training sample
        sample = {
            'prescription_id': prescription_id,
            'image_url': prescription.get('image_url', ''),
            'raw_text': prescription.get('raw_text', ''),
            
            # Ground truth labels
            'patient_name': ground_truth.get('patient', {}).get('name', ''),
            'patient_age': ground_truth.get('patient', {}).get('age', ''),
            'patient_gender': ground_truth.get('patient', {}).get('gender', ''),
            
            'doctor_name': ground_truth.get('doctor', {}).get('name', ''),
            'doctor_specialization': ground_truth.get('doctor', {}).get('specialization', ''),
            
            'medicines': json.dumps(ground_truth.get('medicines', [])),
            'medicine_count': len(ground_truth.get('medicines', [])),
            
            # Metadata
            'original_confidence': prescription.get('confidence', 0),
            'feedback_timestamp': feedback['timestamp'],
            'was_corrected': feedback['feedback_type'] == 'correction'
        }
        
        return sample
    
    def build_ner_dataset(self) -> Tuple[List[Dict], int]:
        """Build NER (Named Entity Recognition) training dataset"""
        logger.info("Building NER dataset...")
        
        # Load training dataset
        latest_dataset = self.training_path / 'latest_dataset.csv'
        if not latest_dataset.exists():
            logger.warning("No training dataset found. Building first...")
            self.build_training_dataset()
        
        df = pd.read_csv(latest_dataset)
        
        ner_samples = []
        
        for _, row in df.iterrows():
            # Create NER training samples from text and labels
            text = row['raw_text']
            
            # Create entity annotations
            entities = self._extract_entities_from_labels(row)
            
            if entities:
                ner_sample = {
                    'text': text,
                    'entities': entities,
                    'prescription_id': row['prescription_id']
                }
                ner_samples.append(ner_sample)
        
        # Save NER dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ner_path = self.training_path / f'ner_dataset_{timestamp}.json'
        
        with open(ner_path, 'w') as f:
            json.dump(ner_samples, f, indent=2)
        
        # Save latest
        latest_ner = self.training_path / 'latest_ner_dataset.json'
        with open(latest_ner, 'w') as f:
            json.dump(ner_samples, f, indent=2)
        
        logger.info(f"Saved NER dataset with {len(ner_samples)} samples")
        
        return ner_samples, len(ner_samples)
    
    def _extract_entities_from_labels(self, row: pd.Series) -> List[Dict]:
        """Extract entity positions from text based on labels"""
        entities = []
        text = row['raw_text']
        
        # Patient name
        if pd.notna(row['patient_name']) and row['patient_name']:
            start = text.find(row['patient_name'])
            if start != -1:
                entities.append({
                    'start': start,
                    'end': start + len(row['patient_name']),
                    'label': 'PATIENT_NAME',
                    'text': row['patient_name']
                })
        
        # Patient age
        if pd.notna(row['patient_age']) and row['patient_age']:
            age_str = str(row['patient_age'])
            start = text.find(age_str)
            if start != -1:
                entities.append({
                    'start': start,
                    'end': start + len(age_str),
                    'label': 'PATIENT_AGE',
                    'text': age_str
                })
        
        # Doctor name
        if pd.notna(row['doctor_name']) and row['doctor_name']:
            start = text.find(row['doctor_name'])
            if start != -1:
                entities.append({
                    'start': start,
                    'end': start + len(row['doctor_name']),
                    'label': 'DOCTOR_NAME',
                    'text': row['doctor_name']
                })
        
        # Medicines
        if pd.notna(row['medicines']):
            medicines = json.loads(row['medicines'])
            for med in medicines:
                med_name = med.get('name', '')
                if med_name:
                    start = text.find(med_name)
                    if start != -1:
                        entities.append({
                            'start': start,
                            'end': start + len(med_name),
                            'label': 'MEDICINE_NAME',
                            'text': med_name
                        })
        
        return entities
    
    def augment_dataset(self, augmentation_factor: int = 2) -> int:
        """Augment training dataset with variations"""
        logger.info(f"Augmenting dataset with factor {augmentation_factor}...")
        
        latest_dataset = self.training_path / 'latest_dataset.csv'
        if not latest_dataset.exists():
            logger.error("No training dataset to augment")
            return 0
        
        df = pd.read_csv(latest_dataset)
        augmented_samples = []
        
        for _, row in df.iterrows():
            # Add original sample
            augmented_samples.append(row.to_dict())
            
            # Create augmented versions
            for i in range(augmentation_factor - 1):
                augmented = self._augment_sample(row, i)
                if augmented:
                    augmented_samples.append(augmented)
        
        # Save augmented dataset
        augmented_df = pd.DataFrame(augmented_samples)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        augmented_path = self.training_path / f'augmented_dataset_{timestamp}.csv'
        augmented_df.to_csv(augmented_path, index=False)
        
        logger.info(f"Created {len(augmented_samples)} augmented samples")
        
        return len(augmented_samples)
    
    def _augment_sample(self, row: pd.Series, version: int) -> Dict:
        """Create an augmented version of a sample"""
        # Text augmentation techniques
        text = row['raw_text']
        
        augmentations = [
            self._add_noise_to_text,
            self._vary_spacing,
            self._add_typos
        ]
        
        # Apply augmentation
        if version < len(augmentations):
            augmented_text = augmentations[version](text)
        else:
            augmented_text = text
        
        augmented = row.to_dict()
        augmented['raw_text'] = augmented_text
        augmented['prescription_id'] = f"{row['prescription_id']}_aug_{version}"
        
        return augmented
    
    def _add_noise_to_text(self, text: str) -> str:
        """Add OCR-like noise to text"""
        # Simulate common OCR errors
        replacements = {
            'o': '0', 'O': '0',
            'l': '1', 'I': '1',
            'S': '5', 's': '5'
        }
        
        import random
        chars = list(text)
        num_changes = max(1, len(chars) // 20)  # Change 5% of characters
        
        for _ in range(num_changes):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] in replacements:
                chars[idx] = replacements[chars[idx]]
        
        return ''.join(chars)
    
    def _vary_spacing(self, text: str) -> str:
        """Vary spacing in text"""
        import re
        # Add/remove extra spaces randomly
        text = re.sub(r'\s+', ' ', text)  # Normalize first
        words = text.split()
        
        import random
        result = []
        for word in words:
            result.append(word)
            if random.random() < 0.1:  # 10% chance of extra space
                result.append(' ')
        
        return ' '.join(result)
    
    def _add_typos(self, text: str) -> str:
        """Add realistic typos"""
        import random
        
        typo_patterns = [
            ('tion', 'toin'),
            ('ing', 'ign'),
            ('the', 'teh'),
        ]
        
        for pattern, typo in typo_patterns:
            if random.random() < 0.3 and pattern in text:
                text = text.replace(pattern, typo, 1)
        
        return text
    
    def export_for_annotation(self, num_samples: int = 50) -> Path:
        """Export samples for manual annotation"""
        logger.info(f"Exporting {num_samples} samples for annotation...")
        
        # Get unannotated prescriptions
        prescriptions = list(self.db['prescriptions'].find({
            'status': 'pending_verification'
        }).limit(num_samples))
        
        annotation_data = []
        
        for prescription in prescriptions:
            annotation_data.append({
                'prescription_id': prescription['prescription_id'],
                'image_url': prescription['image_url'],
                'raw_text': prescription['raw_text'],
                'extracted_data': {
                    'patient': prescription['patient'],
                    'doctor': prescription['doctor'],
                    'medicines': prescription['medicines']
                },
                'confidence': prescription['confidence']
            })
        
        # Save for annotation
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        annotation_path = self.data_path / 'annotations' / f'to_annotate_{timestamp}.json'
        annotation_path.parent.mkdir(exist_ok=True)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        logger.info(f"Exported {len(annotation_data)} samples to {annotation_path}")
        
        return annotation_path
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training dataset"""
        latest_dataset = self.training_path / 'latest_dataset.csv'
        
        if not latest_dataset.exists():
            return {'error': 'No training dataset found'}
        
        df = pd.read_csv(latest_dataset)
        
        stats = {
            'total_samples': len(df),
            'corrected_samples': len(df[df['was_corrected'] == True]),
            'confirmed_samples': len(df[df['was_corrected'] == False]),
            'avg_confidence': df['original_confidence'].mean(),
            'avg_medicines_per_prescription': df['medicine_count'].mean(),
            'date_range': {
                'start': df['feedback_timestamp'].min(),
                'end': df['feedback_timestamp'].max()
            },
            'patient_name_coverage': (df['patient_name'].notna() & (df['patient_name'] != '')).sum() / len(df) * 100,
            'doctor_name_coverage': (df['doctor_name'].notna() & (df['doctor_name'] != '')).sum() / len(df) * 100,
            'medicines_coverage': (df['medicine_count'] > 0).sum() / len(df) * 100
        }
        
        return stats

def main():
    """Main function for dataset building"""
    builder = DatasetBuilder()
    
    # Build training dataset
    df, num_samples = builder.build_training_dataset()
    print(f"✓ Built training dataset with {num_samples} samples")
    
    # Build NER dataset
    ner_samples, num_ner = builder.build_ner_dataset()
    print(f"✓ Built NER dataset with {num_ner} samples")
    
    # Augment dataset
    num_augmented = builder.augment_dataset(augmentation_factor=3)
    print(f"✓ Created {num_augmented} augmented samples")
    
    # Get statistics
    stats = builder.get_dataset_statistics()
    print("\n📊 Dataset Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Export for annotation
    annotation_path = builder.export_for_annotation(num_samples=20)
    print(f"\n📝 Exported samples for annotation: {annotation_path}")

if __name__ == "__main__":
    main()