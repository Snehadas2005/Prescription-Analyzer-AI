#!/usr/bin/env python3
"""
Updated Training Script for Enhanced Prescription Analyzer
Run this script to train and evaluate your analyzer on sample prescription data
"""

import os
import json
import logging
from typing import List, Dict
from datetime import datetime

# Make sure to import your enhanced analyzer
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
except ImportError as e:
    print(f"Error importing EnhancedPrescriptionAnalyzer: {e}")
    print("Make sure prescription_analyzer.py is in the same directory")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrescriptionTrainer:
    def __init__(self, analyzer: EnhancedPrescriptionAnalyzer):
        self.analyzer = analyzer
        
    def get_sample_image_paths(self) -> List[str]:
        """
        Get paths to sample prescription images
        """
        # Define the sample images directory
        samples_dir = "sample_prescriptions"
        
        # Create the directory if it doesn't exist
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
            logger.info(f"Created directory: {samples_dir}")
            
        # Look for prescription images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_paths = []
        
        if os.path.exists(samples_dir):
            for filename in os.listdir(samples_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(samples_dir, filename))
        
        # If no images found, check current directory
        if not image_paths:
            logger.info("No images found in sample_prescriptions/, checking current directory...")
            for filename in os.listdir('.'):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    if 'prescription' in filename.lower() or 'rx' in filename.lower():
                        image_paths.append(filename)
        
        # Sort the paths for consistent ordering
        image_paths.sort()
        
        logger.info(f"Found {len(image_paths)} prescription images")
        for path in image_paths:
            logger.info(f"  - {path}")
            
        return image_paths
        
    def create_training_data_from_samples(self) -> List[Dict]:
        """
        Create training data based on the prescription images you provided
        This includes expected doctor/patient information
        """
        training_data = [
            {
                "sample_id": "prescription_1",
                "description": "Dr. Mandal prescription with patient details",
                "expected_doctor": {
                    "name": "Dr. (Prof.) D.K. Mandal",
                    "specialization": "Professor", 
                    "registration_number": "MBBS, O.O.O+ HD"
                },
                "expected_patient": {
                    "name": "Sunita Mehta",
                    "age": None,
                    "gender": ""
                },
                "expected_medicines": [
                    {
                        "name": "Medicine from prescription",
                        "dosage": "As prescribed",
                        "frequency": "As prescribed", 
                        "duration": "As prescribed"
                    }
                ],
                "notes": "Formal prescription pad with clear doctor credentials"
            },
            {
                "sample_id": "prescription_2",
                "description": "Handwritten prescription with neurologist header",
                "expected_doctor": {
                    "name": "Neurologist name from letterhead",
                    "specialization": "Neurologist",
                    "registration_number": ""
                },
                "expected_patient": {
                    "name": "Patient name if visible",
                    "age": None,
                    "gender": ""
                },
                "expected_medicines": [
                    {
                        "name": "Handwritten medicine names",
                        "dosage": "",
                        "frequency": "",
                        "duration": ""
                    }
                ],
                "notes": "Handwritten prescription, challenging OCR case"
            },
            {
                "sample_id": "prescription_3", 
                "description": "Clinical prescription with doctor information",
                "expected_doctor": {
                    "name": "Dr. Sachin Mehta",
                    "specialization": "Consultant",
                    "registration_number": ""
                },
                "expected_patient": {
                    "name": "Patient name from prescription",
                    "age": None,
                    "gender": ""
                },
                "expected_medicines": [
                    {
                        "name": "Gan Vi",
                        "dosage": "32",
                        "frequency": "bd",
                        "duration": "as prescribed"
                    }
                ],
                "notes": "Mixed handwritten and printed elements"
            },
            {
                "sample_id": "prescription_4",
                "description": "Dr. Abhishek Dubey prescription",
                "expected_doctor": {
                    "name": "Dr. Abhishek Dubey",
                    "specialization": "M.B.B.S., M.D.",
                    "registration_number": ""
                },
                "expected_patient": {
                    "name": "Patient name if visible",
                    "age": None,
                    "gender": ""
                },
                "expected_medicines": [
                    {
                        "name": "Multiple medications from prescription",
                        "dosage": "",
                        "frequency": "",
                        "duration": ""
                    }
                ],
                "notes": "Professional letterhead with clear doctor credentials"
            },
            {
                "sample_id": "prescription_5",
                "description": "Additional prescription sample",
                "expected_doctor": {
                    "name": "Doctor name from prescription",
                    "specialization": "",
                    "registration_number": ""
                },
                "expected_patient": {
                    "name": "Patient name if visible", 
                    "age": None,
                    "gender": ""
                },
                "expected_medicines": [
                    {
                        "name": "Medicine names from prescription",
                        "dosage": "",
                        "frequency": "",
                        "duration": ""
                    }
                ],
                "notes": "Additional sample for comprehensive training"
            }
        ]
        
        return training_data

    def analyze_sample_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Analyze sample prescription images and return results
        """
        results = []
        
        if not image_paths:
            logger.warning("No image paths provided for analysis")
            return results
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Analyzing sample prescription {i+1}/{len(image_paths)}: {image_path}")
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            try:
                # Analyze the prescription
                result = self.analyzer.analyze_prescription(image_path)
                
                analysis_result = {
                    "image_path": image_path,
                    "success": result.success,
                    "prescription_id": result.prescription_id,
                    "extracted_doctor": {
                        "name": result.doctor.name,
                        "specialization": result.doctor.specialization,
                        "registration_number": result.doctor.registration_number
                    },
                    "extracted_patient": {
                        "name": result.patient.name,
                        "age": result.patient.age,
                        "gender": result.patient.gender
                    },
                    "extracted_medicines": [
                        {
                            "name": med.name,
                            "dosage": med.dosage,
                            "frequency": med.frequency,
                            "duration": med.duration
                        } for med in result.medicines
                    ],
                    "confidence_score": result.confidence_score,
                    "raw_text": result.raw_text[:500] + "..." if len(result.raw_text) > 500 else result.raw_text,
                    "error": result.error if not result.success else None
                }
                
                results.append(analysis_result)
                
                # Log key findings
                logger.info(f"✓ Doctor detected: '{result.doctor.name}'")
                logger.info(f"✓ Patient detected: '{result.patient.name}'")
                logger.info(f"✓ Medicines count: {len(result.medicines)}")
                if result.medicines:
                    for med in result.medicines[:3]:  # Show first 3 medicines
                        logger.info(f"  - {med.name} ({med.dosage})")
                logger.info(f"✓ Confidence: {result.confidence_score:.2f}")
                logger.info("-" * 60)
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results

    def generate_training_report(self, results: List[Dict], expected_data: List[Dict]) -> Dict:
        """
        Generate a comprehensive training report
        """
        # Create outputs directory if it doesn't exist
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
            
        successful_results = [r for r in results if r.get("success", False)]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analyzer_version": "Enhanced Prescription Analyzer v1.0",
            "summary": {
                "total_samples": len(results),
                "successful_analyses": len(successful_results),
                "success_rate": len(successful_results) / len(results) if results else 0,
                "average_confidence": sum(r.get("confidence_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
                "total_medicines_detected": sum(len(r.get("extracted_medicines", [])) for r in successful_results),
                "average_medicines_per_prescription": sum(len(r.get("extracted_medicines", [])) for r in successful_results) / len(successful_results) if successful_results else 0
            },
            "detailed_results": results,
            "doctor_patient_analysis": self._analyze_doctor_patient_detection(results),
            "medicine_analysis": self._analyze_medicine_detection(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report
        output_file = os.path.join("outputs", f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            
        logger.info(f"Training report saved to: {output_file}")
        return report

    def _analyze_doctor_patient_detection(self, results: List[Dict]) -> Dict:
        """
        Analyze doctor and patient detection performance
        """
        successful_results = [r for r in results if r.get("success", False)]
        
        doctor_detection = {
            "doctors_detected": sum(1 for r in successful_results if r.get("extracted_doctor", {}).get("name")),
            "doctors_with_specialization": sum(1 for r in successful_results if r.get("extracted_doctor", {}).get("specialization")),
            "doctors_with_license": sum(1 for r in successful_results if r.get("extracted_doctor", {}).get("registration_number")),
            "detection_rate": 0
        }
        
        patient_detection = {
            "patients_detected": sum(1 for r in successful_results if r.get("extracted_patient", {}).get("name")),
            "patients_with_age": sum(1 for r in successful_results if r.get("extracted_patient", {}).get("age")),
            "patients_with_gender": sum(1 for r in successful_results if r.get("extracted_patient", {}).get("gender")),
            "detection_rate": 0
        }
        
        if successful_results:
            doctor_detection["detection_rate"] = doctor_detection["doctors_detected"] / len(successful_results)
            patient_detection["detection_rate"] = patient_detection["patients_detected"] / len(successful_results)
        
        return {
            "doctor_detection": doctor_detection,
            "patient_detection": patient_detection
        }

    def _analyze_medicine_detection(self, results: List[Dict]) -> Dict:
        """
        Analyze medicine detection performance
        """
        successful_results = [r for r in results if r.get("success", False)]
        
        all_medicines = []
        for result in successful_results:
            all_medicines.extend(result.get("extracted_medicines", []))
        
        medicine_analysis = {
            "total_medicines": len(all_medicines),
            "medicines_with_dosage": sum(1 for med in all_medicines if med.get("dosage")),
            "medicines_with_frequency": sum(1 for med in all_medicines if med.get("frequency")),
            "medicines_with_duration": sum(1 for med in all_medicines if med.get("duration")),
            "average_per_prescription": len(all_medicines) / len(successful_results) if successful_results else 0,
            "unique_medicines": list(set(med.get("name", "").lower() for med in all_medicines if med.get("name")))
        }
        
        return medicine_analysis

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """
        Generate recommendations based on results
        """
        recommendations = []
        successful_results = [r for r in results if r.get("success", False)]
        
        success_rate = len(successful_results) / len(results) if results else 0
        
        if success_rate < 0.8:
            recommendations.append("Consider improving image preprocessing for better OCR accuracy")
            
        if successful_results:
            avg_confidence = sum(r.get("confidence_score", 0) for r in successful_results) / len(successful_results)
            if avg_confidence < 0.6:
                recommendations.append("Low confidence scores indicate need for better text extraction methods")
                
            doctor_detection_rate = sum(1 for r in successful_results if r.get("extracted_doctor", {}).get("name")) / len(successful_results)
            if doctor_detection_rate < 0.7:
                recommendations.append("Improve doctor name detection patterns and medical title recognition")
                
            patient_detection_rate = sum(1 for r in successful_results if r.get("extracted_patient", {}).get("name")) / len(successful_results)
            if patient_detection_rate < 0.5:
                recommendations.append("Enhance patient name extraction and context recognition")
                
            avg_medicines = sum(len(r.get("extracted_medicines", [])) for r in successful_results) / len(successful_results)
            if avg_medicines < 1.5:
                recommendations.append("Improve medicine name extraction from handwritten prescriptions")
        
        if not recommendations:
            recommendations.append("Performance is satisfactory. Consider testing on more diverse prescription formats.")
            
        return recommendations


def main():
    """
    Main training and evaluation function
    """
    print("\n" + "="*80)
    print("ENHANCED PRESCRIPTION ANALYZER TRAINING")
    print("="*80)
    
    try:
        # Initialize the analyzer (allowing fallback if no API key)
        logger.info("Initializing Enhanced Prescription Analyzer...")
        analyzer = EnhancedPrescriptionAnalyzer(force_api=False)
        trainer = PrescriptionTrainer(analyzer)
        
        # Get sample image paths
        logger.info("Looking for sample prescription images...")
        image_paths = trainer.get_sample_image_paths()
        
        if not image_paths:
            print("\n❌ NO PRESCRIPTION IMAGES FOUND!")
            print("\nTo use this training script:")
            print("1. Create a folder called 'sample_prescriptions'")
            print("2. Add your prescription images to this folder")
            print("3. Supported formats: .jpg, .jpeg, .png, .tiff, .bmp")
            print("\nAlternatively, place images with 'prescription' in the filename in the current directory.")
            return
        
        # Create expected training data
        expected_data = trainer.create_training_data_from_samples()
        
        # Analyze sample images
        logger.info(f"Starting analysis of {len(image_paths)} prescription images...")
        results = trainer.analyze_sample_images(image_paths)
        
        # Generate comprehensive report
        logger.info("Generating training report...")
        report = trainer.generate_training_report(results, expected_data)
        
        # Print summary to console
        print("\n" + "="*80)
        print("TRAINING RESULTS SUMMARY")
        print("="*80)
        
        summary = report["summary"]
        print(f"📊 Total samples analyzed: {summary['total_samples']}")
        print(f"✅ Successful analyses: {summary['successful_analyses']}")
        print(f"📈 Success rate: {summary['success_rate']*100:.1f}%")
        print(f"🎯 Average confidence: {summary['average_confidence']*100:.1f}%")
        print(f"💊 Total medicines detected: {summary['total_medicines_detected']}")
        print(f"📋 Average medicines per prescription: {summary['average_medicines_per_prescription']:.1f}")
        
        # Doctor/Patient detection analysis
        dp_analysis = report["doctor_patient_analysis"]
        print(f"\n👨‍⚕️ Doctor detection rate: {dp_analysis['doctor_detection']['detection_rate']*100:.1f}%")
        print(f"👤 Patient detection rate: {dp_analysis['patient_detection']['detection_rate']*100:.1f}%")
        
        print(f"\n📝 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
            
        print(f"\n📄 Detailed report saved to: outputs/training_report_*.json")
        print("="*80)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()