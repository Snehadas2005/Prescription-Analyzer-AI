#!/bin/bash

echo "íº€ Starting All Services"
echo "========================"
echo ""

# Kill any existing processes on ports 3000, 8000, 8080
echo "í·¹ Cleaning up existing processes..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 2

# Start ML Service
echo "í´– Starting ML Service on port 8000..."
cd ml-service
cp app/main_fixed.py app/main.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/ml-service.log 2>&1 &
ML_PID=$!
cd ..
sleep 5

# Check ML Service
if curl -s http://localhost:8000/health > /dev/null; then
    echo "âœ… ML Service started successfully"
else
    echo "âŒ ML Service failed to start"
    exit 1
fi

# Start Go Backend
echo "í´§ Starting Go Backend on port 8080..."
cd backend
go run cmd/server/main.go > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3

# Check Go Backend
if curl -s http://localhost:8080/health > /dev/null; then
    echo "âœ… Go Backend started successfully"
else
    echo "âŒ Go Backend failed to start"
    exit 1
fi

# Start React Frontend
echo "âš›ï¸  Starting React Frontend on port 3000..."
cd frontend
BROWSER=none npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "âœ… All services started!"
echo ""
echo "Service URLs:"
echo "- Frontend:    http://localhost:3000"
echo "- Go Backend:  http://localhost:8080"
echo "- ML Service:  http://localhost:8000"
echo "- ML Docs:     http://localhost:8000/docs"
echo ""
echo "Process IDs:"
echo "- ML Service: $ML_PID"
echo "- Backend:    $BACKEND_PID"
echo "- Frontend:   $FRONTEND_PID"
echo ""
echo "Logs available at:"
echo "- logs/ml-service.log"
echo "- logs/backend.log"
echo "- logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup on exit
trap "echo 'Stopping all services...'; kill $ML_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" EXIT INT TERM

# Wait
wait
