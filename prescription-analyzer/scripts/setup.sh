#!/bin/bash

echo "🚀 Setting up AI Prescription Analyzer..."

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v go >/dev/null 2>&1 || { echo "❌ Go is required but not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }

echo "✅ All prerequisites found"

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Setup Frontend
echo "📦 Setting up Frontend..."
cd frontend
npm install
cd ..

# Setup Backend
echo "📦 Setting up Backend..."
cd backend
go mod download
cd ..

# Setup ML Service
echo "📦 Setting up ML Service..."
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
deactivate
cd ..

# Setup Training Pipeline
echo "📦 Setting up Training Pipeline..."
cd training-pipeline
pip install -r requirements.txt
cd ..

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ml-service/data/{raw,processed,training,feedback,feedback/details}
mkdir -p ml-service/models/{production,checkpoints}
mkdir -p ml-service/logs
mkdir -p backend/uploads
mkdir -p backend/logs

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Start MongoDB: docker-compose up mongo -d"
echo "3. Start all services: docker-compose up"
echo "   OR run individually:"
echo "   - Backend: cd backend && go run cmd/server/main.go"
echo "   - ML Service: cd ml-service && uvicorn app.main:app --reload"
echo "   - Frontend: cd frontend && npm start"

# File: scripts/start-dev.sh
#!/bin/bash

echo "🚀 Starting Development Environment..."

# Start MongoDB
echo "📦 Starting MongoDB..."
docker-compose up -d mongo

# Wait for MongoDB to be ready
echo "⏳ Waiting for MongoDB..."
sleep 5

# Start Backend in background
echo "🔧 Starting Backend..."
cd backend
go run cmd/server/main.go &
BACKEND_PID=$!
cd ..

# Start ML Service in background
echo "🤖 Starting ML Service..."
cd ml-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 &
ML_PID=$!
deactivate
cd ..

# Wait a bit for services to start
sleep 5

# Start Frontend
echo "⚛️  Starting Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "✅ All services started!"
echo ""
echo "Services:"
echo "- Frontend: http://localhost:3000"
echo "- Backend: http://localhost:8080"
echo "- ML Service: http://localhost:8000"
echo "- MongoDB: mongodb://localhost:27017"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup on exit
trap "echo 'Stopping services...'; kill $BACKEND_PID $ML_PID $FRONTEND_PID; docker-compose stop mongo" EXIT

# Wait for user interrupt
wait

# File: scripts/deploy.sh
#!/bin/bash

echo "🚀 Deploying AI Prescription Analyzer..."

# Build Frontend
echo "📦 Building Frontend..."
cd frontend
npm run build
cd ..

# Deploy to Netlify
if command -v netlify >/dev/null 2>&1; then
    echo "🌐 Deploying to Netlify..."
    cd frontend
    netlify deploy --prod --dir=build
    cd ..
else
    echo "⚠️  Netlify CLI not found. Install with: npm install -g netlify-cli"
fi

# Build Backend Docker image
echo "🐳 Building Backend Docker image..."
cd backend
docker build -t prescription-backend .
cd ..

# Build ML Service Docker image
echo "🐳 Building ML Service Docker image..."
cd ml-service
docker build -t prescription-ml-service .
cd ..

echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "1. Push Docker images to registry"
echo "2. Deploy backend to Railway/Render"
echo "3. Deploy ML service to Railway/Render"
echo "4. Update environment variables in production"