#!/bin/bash

# AI Prescription Analyzer - Complete Startup Script
# This script starts all services in the correct order

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Cleanup function
cleanup() {
    print_warning "Shutting down services..."
    
    # Kill all background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    
    print_success "All services stopped"
    exit 0
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Start script
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     AI Prescription Analyzer - Complete Startup           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

if ! command -v go &> /dev/null; then
    print_error "Go is required but not installed"
    exit 1
fi

if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    exit 1
fi

print_success "All prerequisites found"

# Check if ports are available
print_status "Checking if ports are available..."

if check_port 3000; then
    print_error "Port 3000 (Frontend) is already in use"
    print_warning "Kill the process using: lsof -ti:3000 | xargs kill -9"
    exit 1
fi

if check_port 8080; then
    print_error "Port 8080 (Backend) is already in use"
    print_warning "Kill the process using: lsof -ti:8080 | xargs kill -9"
    exit 1
fi

if check_port 8000; then
    print_error "Port 8000 (ML Service) is already in use"
    print_warning "Kill the process using: lsof -ti:8000 | xargs kill -9"
    exit 1
fi

print_success "All ports are available"

# Set environment variables
print_status "Setting up environment variables..."

# Check if .env exists in backend
if [ ! -f "backend/.env" ]; then
    print_warning ".env file not found in backend/"
    print_status "Creating .env file with default values..."
    cat > backend/.env << EOF
# ML Service Configuration
ML_SERVICE_URL=http://localhost:8000

# Cohere API (optional - will use pattern matching if not provided)
# Get your API key from: https://dashboard.cohere.com/api-keys
COHERE_API_KEY=

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=prescription_analyzer

# Server Configuration
PORT=8080
HOST=0.0.0.0
EOF
    print_success "Created backend/.env file"
    print_warning "Please add your COHERE_API_KEY to backend/.env for better results"
fi

# Export ML_SERVICE_URL for the backend
export ML_SERVICE_URL="http://localhost:8000"

print_success "Environment variables configured"

# Step 1: Start ML Service (Python FastAPI)
print_status "Starting ML Service on port 8000..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv312" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv312
fi

# Activate virtual environment and start service
source venv312/bin/activate

# Install requirements if needed
if ! python -c "import fastapi" 2>/dev/null; then
    print_status "Installing Python dependencies..."
    pip install -r integration/requirements.txt
fi

# Start ML service in background
print_status "Launching ML service..."
python main.py > ../logs/ml-service.log 2>&1 &
ML_PID=$!

deactivate
cd ..

# Wait for ML service to be ready
if ! wait_for_service "http://localhost:8000/health" "ML Service"; then
    print_error "ML Service failed to start. Check logs/ml-service.log"
    exit 1
fi

# Step 2: Start Backend (Go)
print_status "Starting Backend service on port 8080..."

cd backend

# Build and start backend
print_status "Building Go backend..."
go build -o bin/server cmd/server/main.go

print_status "Launching backend service..."
./bin/server > ../logs/backend.log 2>&1 &
BACKEND_PID=$!

cd ..

# Wait for backend to be ready
if ! wait_for_service "http://localhost:8080/health" "Backend"; then
    print_error "Backend failed to start. Check logs/backend.log"
    exit 1
fi

# Step 3: Start Frontend (React)
print_status "Starting Frontend on port 3000..."

cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
fi

# Set environment variable for API URL
export REACT_APP_API_URL="http://localhost:8080/api/v1"

print_status "Launching frontend..."
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!

cd ..

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:3000" "Frontend"; then
    print_error "Frontend failed to start. Check logs/frontend.log"
    exit 1
fi

# All services started successfully
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              🎉 All Services Running! 🎉                   ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
print_success "Frontend:    http://localhost:3000"
print_success "Backend:     http://localhost:8080"
print_success "ML Service:  http://localhost:8000"
print_success "API Docs:    http://localhost:8000/docs"
echo ""
print_status "Process IDs:"
echo "  - ML Service: $ML_PID"
echo "  - Backend:    $BACKEND_PID"
echo "  - Frontend:   $FRONTEND_PID"
echo ""
print_status "Logs are being written to:"
echo "  - logs/ml-service.log"
echo "  - logs/backend.log"
echo "  - logs/frontend.log"
echo ""
print_warning "Press Ctrl+C to stop all services"
echo ""

# Tail all logs
print_status "Tailing service logs (Ctrl+C to stop)..."
tail -f logs/*.log

# Wait for user to stop
wait