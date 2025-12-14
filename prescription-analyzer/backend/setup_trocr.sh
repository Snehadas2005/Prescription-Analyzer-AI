#!/bin/bash

set -e

echo "=============================================================================="
echo "  TrOCR INTEGRATION SETUP - Handwritten Prescription Support"
echo "=============================================================================="

# Check if running from correct directory
if [ ! -f "prescription_analyzer.py" ]; then
    echo "❌ Error: prescription_analyzer.py not found"
    echo "Please run this script from the backend directory"
    exit 1
fi

echo "Step 1: Checking Python environment..."
# Check Python version and activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv312" ]; then
        source venv312/bin/activate
    else
        python3 -m venv venv312
        source venv312/bin/activate
    fi
fi

echo "Step 2: Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.35.2
pip install sentencepiece==0.1.99

echo "Step 3: Downloading TrOCR model..."
python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print('Downloading TrOCR processor...')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
print('Downloading TrOCR model...')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print('✅ TrOCR MODEL READY!')
"

echo "✅ SETUP COMPLETE!"
echo "Next steps:"
echo "  1. Test on a prescription: python prescription_analyzer.py your_prescription.jpg"
echo "  2. Compare OCR methods: python test_trocr.py your_prescription.jpg"