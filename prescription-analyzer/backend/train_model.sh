#!/bin/bash

# Training Script for Self-Learning Prescription Analyzer

echo "======================================================================"
echo "  PRESCRIPTION ANALYZER - SELF-LEARNING TRAINING"
echo "======================================================================"
echo ""

# Check for data folder
if [ ! -d "../data" ]; then
    echo "❌ Error: data/prescriptions folder not found"
    echo "Please create it and add 142 prescription images"
    exit 1
fi

# Count images
image_count=$(find ../data -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
echo "Found $image_count prescription images"

if [ $image_count -lt 10 ]; then
    echo "❌ Not enough training data (need at least 10, prefer 142+)"
    exit 1
fi

# Activate virtual environment
if [ -d "venv312" ]; then
    source venv312/Scripts/activate
else
    echo "❌ Virtual environment not found"
    exit 1
fi

echo ""
echo "Starting training process..."
echo ""

# Run training
python enhanced_trainer.py

echo ""
echo "======================================================================"
echo "  TRAINING COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check models/trained/ for generated knowledge bases"
echo "  2. Test with: python test_analyzer.py path/to/test/prescription.jpg"
echo "  3. Restart ML service to use new models"
echo ""