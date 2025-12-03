#!/bin/bash

# Debug Script for AI Prescription Analyzer
# This script helps diagnose common issues

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_check() {
    if [ $2 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
    fi
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     AI Prescription Analyzer - Debug & Diagnostic         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"

# 1. Check Prerequisites
print_header "1. Checking Prerequisites"

command -v python3 >/dev/null 2>&1
print_check "Python 3 installed" $?
if command -v python3 >/dev/null 2>&1; then
    echo "   Version: $(python3 --version)"
fi

command -v go >/dev/null 2>&1
print_check "Go installed" $?
if command -v go >/dev/null 2>&1; then
    echo "   Version: $(go version)"
fi

command -v node >/dev/null 2>&1
print_check "Node.js installed" $?
if command -v node >/dev/null 2>&1; then
    echo "   Version: $(node --version)"
fi

command -v npm >/dev/null 2>&1
print_check "npm installed" $?

# 2. Check Directory Structure
print_header "2. Checking Directory Structure"

directories=(
    "backend"
    "backend/cmd/server"
    "backend/internal"
    "frontend"
    "frontend/src"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        print_check "$dir exists" 0
    else
        print_check "$dir exists" 1
    fi
done

# 3. Check Important Files
print_header "3. Checking Important Files"

files=(
    "backend/main.py"
    "backend/prescription_analyzer.py"
    "backend/cmd/server/main.go"
    "frontend/package.json"
    "frontend/src/App.js"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        print_check "$file exists" 0
    else
        print_check "$file exists" 1
        echo -e "   ${RED}Missing: $file${NC}"
    fi
done

# 4. Check Environment Files
print_header "4. Checking Environment Configuration"

if [ -f "backend/.env" ]; then
    print_check "backend/.env exists" 0
    echo "   Contents:"
    cat backend/.env | grep -v "API_KEY" | sed 's/^/   /'
else
    print_check "backend/.env exists" 1
    echo -e "   ${YELLOW}Create backend/.env with ML_SERVICE_URL and other configs${NC}"
fi

# 5. Check Port Usage
print_header "5. Checking Port Availability"

ports=(3000 8080 8000)
port_names=("Frontend" "Backend" "ML Service")

for i in "${!ports[@]}"; do
    port=${ports[$i]}
    name=${port_names[$i]}
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        pid=$(lsof -ti:$port)
        process=$(ps -p $pid -o comm=)
        print_check "$name (port $port) available" 1
        echo -e "   ${YELLOW}Port $port is in use by PID $pid ($process)${NC}"
        echo -e "   ${YELLOW}Kill with: kill -9 $pid${NC}"
    else
        print_check "$name (port $port) available" 0
    fi
done

# 6. Check Python Dependencies
print_header "6. Checking Python Dependencies"

if [ -d "backend/venv312" ]; then
    print_check "Python virtual environment exists" 0
    
    # Activate and check packages
    source backend/venv312/bin/activate 2>/dev/null
    
    packages=("fastapi" "uvicorn" "easyocr" "cv2" "PIL" "cohere")
    for pkg in "${packages[@]}"; do
        python -c "import $pkg" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_check "$pkg installed" 0
        else
            print_check "$pkg installed" 1
        fi
    done
    
    deactivate 2>/dev/null
else
    print_check "Python virtual environment exists" 1
    echo -e "   ${YELLOW}Create with: cd backend && python3 -m venv venv312${NC}"
fi

# 7. Check Go Dependencies
print_header "7. Checking Go Dependencies"

if [ -f "backend/go.mod" ]; then
    print_check "go.mod exists" 0
else
    print_check "go.mod exists" 1
fi

if [ -f "backend/go.sum" ]; then
    print_check "go.sum exists" 0
else
    print_check "go.sum exists" 1
fi

# 8. Check Node Dependencies
print_header "8. Checking Node.js Dependencies"

if [ -d "frontend/node_modules" ]; then
    print_check "node_modules exists" 0
else
    print_check "node_modules exists" 1
    echo -e "   ${YELLOW}Install with: cd frontend && npm install${NC}"
fi

# 9. Check Services Status
print_header "9. Checking Services Status"

services=(
    "http://localhost:8000/health|ML Service"
    "http://localhost:8080/health|Backend"
    "http://localhost:3000|Frontend"
)

for service in "${services[@]}"; do
    IFS='|' read -r url name <<< "$service"
    
    if curl -s "$url" > /dev/null 2>&1; then
        print_check "$name is running" 0
        response=$(curl -s "$url" 2>/dev/null)
        if [ -n "$response" ]; then
            echo "   Response: $response" | head -c 100
        fi
    else
        print_check "$name is running" 1
    fi
done

# 10. Check Log Files
print_header "10. Checking Log Files"

if [ -d "logs" ]; then
    print_check "logs directory exists" 0
    
    log_files=(
        "logs/ml-service.log"
        "logs/backend.log"
        "logs/frontend.log"
    )
    
    for log in "${log_files[@]}"; do
        if [ -f "$log" ]; then
            size=$(du -h "$log" | cut -f1)
            lines=$(wc -l < "$log")
            print_check "$log exists" 0
            echo "   Size: $size, Lines: $lines"
            
            # Check for errors in logs
            if grep -q -i "error\|failed\|exception" "$log" 2>/dev/null; then
                echo -e "   ${YELLOW}Contains errors - check with: tail -n 50 $log${NC}"
            fi
        else
            print_check "$log exists" 1
        fi
    done
else
    print_check "logs directory exists" 1
    echo -e "   ${YELLOW}Create with: mkdir -p logs${NC}"
fi

# 11. Test API Endpoints
print_header "11. Testing API Endpoints"

if curl -s "http://localhost:8080/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend is accessible"
    
    # Test upload endpoint
    echo "Testing upload endpoint..."
    response=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/v1/upload 2>/dev/null | tail -1)
    if [ "$response" = "404" ] || [ "$response" = "405" ]; then
        echo -e "${GREEN}✓${NC} Upload endpoint exists (returns $response for GET)"
    else
        echo -e "${YELLOW}⚠${NC} Unexpected response: $response"
    fi
fi

if curl -s "http://localhost:8000/docs" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} ML Service API docs accessible at http://localhost:8000/docs"
fi

# 12. Recommendations
print_header "12. Recommendations"

echo ""
echo "To fix common issues:"
echo ""
echo "1. Install missing dependencies:"
echo "   cd backend && source venv312/bin/activate && pip install -r integration/requirements.txt"
echo ""
echo "2. Kill processes on occupied ports:"
echo "   lsof -ti:8000 | xargs kill -9  # ML Service"
echo "   lsof -ti:8080 | xargs kill -9  # Backend"
echo "   lsof -ti:3000 | xargs kill -9  # Frontend"
echo ""
echo "3. Create logs directory:"
echo "   mkdir -p logs"
echo ""
echo "4. Start services with startup script:"
echo "   chmod +x startup.sh && ./startup.sh"
echo ""
echo "5. View logs:"
echo "   tail -f logs/ml-service.log"
echo "   tail -f logs/backend.log"
echo "   tail -f logs/frontend.log"
echo ""

print_header "Debug Complete"