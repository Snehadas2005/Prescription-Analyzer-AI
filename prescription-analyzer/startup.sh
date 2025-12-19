#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}   AI PRESCRIPTION ANALYZER - COMPLETE STARTUP${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"
MISSING_DEPS=0

if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 is required but not installed${NC}"
    MISSING_DEPS=1
fi

if ! command_exists go; then
    echo -e "${RED}✗ Go is required but not installed${NC}"
    MISSING_DEPS=1
fi

if ! command_exists node; then
    echo -e "${RED}✗ Node.js is required but not installed${NC}"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}❌ Missing required dependencies. Please install them first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# Check Python virtual environment for backend
echo -e "${YELLOW}Step 2: Setting up Python environment...${NC}"
cd backend

if [ ! -d "venv312" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv312
fi

source venv312/bin/activate || source venv312/Scripts/activate

echo "Installing/updating Python dependencies..."
pip install -q --upgrade pip
pip install -q -r integration/requirements.txt

echo -e "${GREEN}✓ Python environment ready${NC}"
cd ..
echo ""

# Check Cohere API Key
echo -e "${YELLOW}Step 3: Checking API configuration...${NC}"
if [ -f "backend/integration/keys.py" ]; then
    echo -e "${GREEN}✓ API keys file found${NC}"
else
    echo -e "${YELLOW}⚠ No API keys file found${NC}"
    echo -e "${YELLOW}  Creating template keys.py...${NC}"
    cat > backend/integration/keys.py << 'EOF'
# Replace with your actual Cohere API key
COHERE_API_KEY = "your-cohere-api-key-here"
EOF
    echo -e "${YELLOW}  Please edit backend/integration/keys.py with your Cohere API key${NC}"
fi
echo ""

# Check if ports are available
echo -e "${YELLOW}Step 4: Checking port availability...${NC}"
PORTS_BUSY=0

if port_in_use 3000; then
    echo -e "${YELLOW}⚠ Port 3000 (Frontend) is already in use${NC}"
    PORTS_BUSY=1
fi

if port_in_use 8080; then
    echo -e "${YELLOW}⚠ Port 8080 (Go Backend) is already in use${NC}"
    PORTS_BUSY=1
fi

if port_in_use 8000; then
    echo -e "${YELLOW}⚠ Port 8000 (Python ML Service) is already in use${NC}"
    PORTS_BUSY=1
fi

if [ $PORTS_BUSY -eq 1 ]; then
    echo -e "${YELLOW}Would you like to kill processes on these ports? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Killing processes on busy ports..."
        lsof -ti:3000 | xargs kill -9 2>/dev/null
        lsof -ti:8080 | xargs kill -9 2>/dev/null
        lsof -ti:8000 | xargs kill -9 2>/dev/null
        sleep 2
        echo -e "${GREEN}✓ Ports cleared${NC}"
    fi
fi
echo ""

# Create necessary directories
echo -e "${YELLOW}Step 5: Creating necessary directories...${NC}"
mkdir -p backend/uploads
mkdir -p backend/logs
mkdir -p ml-service/data/feedback
mkdir -p ml-service/models
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Start services
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}   STARTING SERVICES${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Start Python ML Service
echo -e "${YELLOW}Starting Python ML Service (Port 8000)...${NC}"
cd backend
source venv312/bin/activate || source venv312/Scripts/activate
python main.py > ../logs/ml-service.log 2>&1 &
ML_PID=$!
echo -e "${GREEN}✓ ML Service started (PID: $ML_PID)${NC}"
cd ..
sleep 3

# Check if ML service is running
if ps -p $ML_PID > /dev/null; then
    echo -e "${GREEN}✓ ML Service is running${NC}"
    curl -s http://localhost:8000/health > /dev/null && echo -e "${GREEN}✓ ML Service health check passed${NC}" || echo -e "${YELLOW}⚠ ML Service health check pending...${NC}"
else
    echo -e "${RED}✗ ML Service failed to start. Check logs/ml-service.log${NC}"
fi
echo ""

# Start Go Backend (if needed)
echo -e "${YELLOW}Starting Go Backend (Port 8080)...${NC}"
if [ -f "backend/cmd/server/main.go" ]; then
    cd backend
    go run cmd/server/main.go > ../logs/backend.log 2>&1 &
    GO_PID=$!
    echo -e "${GREEN}✓ Go Backend started (PID: $GO_PID)${NC}"
    cd ..
    sleep 2
else
    echo -e "${YELLOW}⚠ Go backend not found, skipping...${NC}"
    GO_PID=""
fi
echo ""

# Start React Frontend
echo -e "${YELLOW}Starting React Frontend (Port 3000)...${NC}"
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    cd frontend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
    cd ..
    sleep 5
else
    echo -e "${YELLOW}⚠ Frontend not found, skipping...${NC}"
    FRONTEND_PID=""
fi
echo ""

# Summary
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}   STARTUP COMPLETE${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Services Status:${NC}"
echo -e "  ${GREEN}✓${NC} ML Service:       http://localhost:8000"
echo -e "                      Health: http://localhost:8000/health"
echo -e "                      Docs:   http://localhost:8000/docs"

if [ ! -z "$GO_PID" ]; then
    echo -e "  ${GREEN}✓${NC} Go Backend:       http://localhost:8080"
    echo -e "                      Health: http://localhost:8080/health"
fi

if [ ! -z "$FRONTEND_PID" ]; then
    echo -e "  ${GREEN}✓${NC} React Frontend:   http://localhost:3000"
fi

echo ""
echo -e "${YELLOW}Process IDs:${NC}"
echo -e "  ML Service:  $ML_PID"
[ ! -z "$GO_PID" ] && echo -e "  Go Backend:  $GO_PID"
[ ! -z "$FRONTEND_PID" ] && echo -e "  Frontend:    $FRONTEND_PID"
echo ""

echo -e "${YELLOW}Log Files:${NC}"
echo -e "  logs/ml-service.log"
echo -e "  logs/backend.log"
echo -e "  logs/frontend.log"
echo ""

echo -e "${BLUE}Quick Test Commands:${NC}"
echo -e "  # Test ML Service health"
echo -e "  ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo -e "  # Test prescription analysis"
echo -e "  ${GREEN}curl -X POST -F \"file=@your-prescription.jpg\" http://localhost:8000/analyze-prescription${NC}"
echo ""

echo -e "${BLUE}To stop all services:${NC}"
echo -e "  ${RED}kill $ML_PID${NC}"
[ ! -z "$GO_PID" ] && echo -e "  ${RED}kill $GO_PID${NC}"
[ ! -z "$FRONTEND_PID" ] && echo -e "  ${RED}kill $FRONTEND_PID${NC}"
echo ""

echo -e "${BLUE}Or use:${NC}"
echo -e "  ${RED}pkill -f \"python main.py\"${NC}"
echo -e "  ${RED}pkill -f \"go run\"${NC}"
echo -e "  ${RED}pkill -f \"npm start\"${NC}"
echo ""

# Save PIDs to file for easy cleanup
cat > .pids << EOF
ML_PID=$ML_PID
GO_PID=$GO_PID
FRONTEND_PID=$FRONTEND_PID
EOF

echo -e "${GREEN}✨ All services started successfully!${NC}"
echo -e "${YELLOW}Press Ctrl+C to view logs, or run 'tail -f logs/*.log' in another terminal${NC}"
echo ""

# Keep script running and tail logs
trap "echo 'Shutting down...'; [ ! -z '$ML_PID' ] && kill $ML_PID 2>/dev/null; [ ! -z '$GO_PID' ] && kill $GO_PID 2>/dev/null; [ ! -z '$FRONTEND_PID' ] && kill $FRONTEND_PID 2>/dev/null; rm -f .pids; exit" INT TERM

# Wait and show logs
sleep 2
echo -e "${BLUE}Showing ML Service logs (Press Ctrl+C to stop):${NC}"
tail -f logs/ml-service.log