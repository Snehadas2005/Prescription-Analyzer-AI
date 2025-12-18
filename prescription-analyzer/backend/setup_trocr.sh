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
# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "✓ Python version: $python_version"

# Activate or create virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv312" ]; then
        echo "Activating existing virtual environment..."
        source venv312/bin/activate
    else
        echo "Creating new virtual environment..."
        python3 -m venv venv312
        source venv312/bin/activate
    fi
fi

echo "Step 2: Installing PyTorch (CPU version)..."
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

echo "Step 3: Installing Transformers and dependencies..."
pip install transformers==4.35.2
pip install sentencepiece==0.1.99
pip install accelerate==0.25.0
pip install tokenizers==0.15.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1

echo "Step 4: Downloading TrOCR models..."
python3 << 'PYTHON_SCRIPT'
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

print('📥 Downloading TrOCR processor...')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

print('📥 Downloading TrOCR model...')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Test the model
print('🧪 Testing TrOCR model...')
print(f'✓ Processor loaded successfully')
print(f'✓ Model loaded successfully')
print(f'✓ Model device: {next(model.parameters()).device}')
print(f'✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')

print('\n✅ TrOCR MODEL READY!')
PYTHON_SCRIPT

echo "Step 5: Installing self-learning dependencies..."
pip install scikit-learn==1.3.2 joblib==1.3.2

echo ""
echo "✅ SETUP COMPLETE!"
echo ""
echo "Next steps:"
echo "  1. Test on a prescription: python prescription_analyzer.py your_prescription.jpg"
echo "  2. Test hybrid OCR: python test_hybrid_ocr.py your_prescription.jpg"
echo "  3. Train on feedback: python training_script.py"
echo ""
echo "TrOCR model location: ~/.cache/huggingface/hub/"
echo "Model size: ~300MB"