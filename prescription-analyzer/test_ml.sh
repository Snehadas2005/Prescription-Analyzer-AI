#!/bin/bash

# Quick Test Script for ML Service
# Run this AFTER the fix script

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}║                   ${GREEN}ML SERVICE TEST${BLUE}                                ║${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if ML service is running
echo -e "${YELLOW}Checking if ML service is running...${NC}"

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ML Service is running!${NC}"
    echo ""
    
    # Get health status
    echo -e "${YELLOW}Health Check Response:${NC}"
    curl -s http://localhost:8000/health | python -m json.tool
    echo ""
    
    # Get root info
    echo -e "${YELLOW}Root Endpoint Response:${NC}"
    curl -s http://localhost:8000/ | python -m json.tool
    echo ""
    
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ ML SERVICE IS WORKING!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}To test with a real prescription image:${NC}"
    echo -e "${GREEN}curl -X POST -F \"file=@your-prescription.jpg\" http://localhost:8000/analyze-prescription${NC}"
    
else
    echo -e "${RED}✗ ML Service is NOT running${NC}"
    echo ""
    echo -e "${YELLOW}To start the ML service:${NC}"
    echo -e "1. Open a terminal"
    echo -e "2. ${GREEN}cd prescription-analyzer/backend${NC}"
    echo -e "3. ${GREEN}source venv312/Scripts/activate${NC}  (Windows)"
    echo -e "   ${GREEN}source venv312/bin/activate${NC}      (Linux/Mac)"
    echo -e "4. ${GREEN}cd ../ml-service${NC}"
    echo -e "5. ${GREEN}python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload${NC}"
fi

echo ""