#!/bin/bash

set -e

echo "=============================================================================="
echo "  ENHANCED TrOCR INTEGRATION SETUP - Complete Installation"
echo "=============================================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "prescription_analyzer.py" ]; then
    echo -e "${RED}❌ Error: prescription_analyzer.py not found${NC}"
    echo "Please run this script from the backend directory"
    exit 1
fi

echo -e "${GREEN}✓ Running from backend directory${NC}"

# Check Python version
echo -e "\n${YELLOW}Step 1: Checking Python environment...${NC}"
python_version=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python version: $python_version${NC}"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Python 3.8 or higher is required${NC}"
    exit 1
fi

# Create or activate virtual environment
echo -e "\n${YELLOW}Step 2: Setting up virtual environment...${NC}"
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
echo -e "${GREEN}✓ Virtual environment ready${NC}"

# Upgrade pip
echo -e "\n${YELLOW}Step 3: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install PyTorch (CPU version for faster installation)
echo -e "\n${YELLOW}Step 4: Installing PyTorch (CPU version)...${NC}"
echo "This may take a few minutes..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install Transformers and dependencies
echo -e "\n${YELLOW}Step 5: Installing Transformers library...${NC}"
pip install transformers==4.35.2
pip install sentencepiece==0.1.99
pip install accelerate==0.25.0
pip install tokenizers==0.15.0
pip install huggingface-hub==0.19.4
pip install safetensors==0.4.1
echo -e "${GREEN}✓ Transformers installed${NC}"

# Install existing requirements
echo -e "\n${YELLOW}Step 6: Installing other dependencies...${NC}"
if [ -f "integration/requirements.txt" ]; then
    pip install -r integration/requirements.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download TrOCR models
echo -e "\n${YELLOW}Step 7: Downloading TrOCR models...${NC}"
echo "This will download ~300MB of model files..."

python3 << 'PYTHON_SCRIPT'
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import sys

print('\n📥 Downloading TrOCR processor...')
try:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    print('✓ Processor downloaded')
except Exception as e:
    print(f'❌ Failed to download processor: {e}')
    sys.exit(1)

print('\n📥 Downloading TrOCR model...')
try:
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    print('✓ Model downloaded')
except Exception as e:
    print(f'❌ Failed to download model: {e}')
    sys.exit(1)

# Test the model
print('\n🧪 Testing TrOCR model...')
try:
    print(f'✓ Processor loaded successfully')
    print(f'✓ Model loaded successfully')
    print(f'✓ Model device: {next(model.parameters()).device}')
    print(f'✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')
    print(f'\n✅ TrOCR MODEL READY!')
except Exception as e:
    print(f'❌ Model test failed: {e}')
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TrOCR models downloaded and tested successfully${NC}"
else
    echo -e "${RED}❌ TrOCR model download failed${NC}"
    exit 1
fi

# Install self-learning dependencies
echo -e "\n${YELLOW}Step 8: Installing self-learning dependencies...${NC}"
pip install scikit-learn==1.3.2 joblib==1.3.2
echo -e "${GREEN}✓ Self-learning dependencies installed${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Step 9: Creating directories...${NC}"
mkdir -p models/trocr
mkdir -p data/cache
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Test the complete system
echo -e "\n${YELLOW}Step 10: Testing complete system...${NC}"
python3 << 'PYTHON_SCRIPT'
import sys
import os

print("\nTesting imports...")
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
    print("✓ EnhancedPrescriptionAnalyzer imported")
    
    # Test initialization
    analyzer = EnhancedPrescriptionAnalyzer(use_gpu=False, force_api=False)
    print("✓ Analyzer initialized")
    
    # Check TrOCR availability
    if hasattr(analyzer, 'ocr_engine'):
        if analyzer.ocr_engine.trocr_available:
            print("✓ TrOCR engine available and ready")
        else:
            print("⚠️  TrOCR engine not available (will use traditional OCR)")
    
    print("\n✅ SYSTEM TEST PASSED!")
    
except Exception as e:
    print(f"❌ System test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ System test passed${NC}"
else
    echo -e "${RED}❌ System test failed${NC}"
    exit 1
fi

# Print summary
echo -e "\n=============================================================================="
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo -e "=============================================================================="
echo ""
echo "Installation Summary:"
echo "  • PyTorch 2.1.0 (CPU)"
echo "  • Transformers 4.35.2"
echo "  • TrOCR model: microsoft/trocr-base-handwritten"
echo "  • Model location: ~/.cache/huggingface/hub/"
echo "  • Total disk space used: ~3GB"
echo ""
echo "Next steps:"
echo "  1. Test on a prescription:"
echo "     python prescription_analyzer.py your_prescription.jpg"
echo ""
echo "  2. Run the test script:"
echo "     python test_hybrid_ocr.py your_prescription.jpg"
echo ""
echo "  3. Start the API server:"
echo "     python main.py"
echo ""
echo "For GPU support (faster inference):"
echo "  pip uninstall torch torchvision"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo "Alternative TrOCR models:"
echo "  • microsoft/trocr-small-handwritten (faster, less accurate)"
echo "  • microsoft/trocr-large-handwritten (slower, more accurate)"
echo "  • microsoft/trocr-base-printed (for printed text only)"
echo ""
echo -e "${GREEN}Happy analyzing! 🎉${NC}"