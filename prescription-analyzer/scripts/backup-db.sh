#!/bin/bash

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "📦 Creating MongoDB backup..."
docker exec prescription_mongo mongodump --out /tmp/backup

docker cp prescription_mongo:/tmp/backup $BACKUP_DIR/backup_$TIMESTAMP

echo "✅ Backup created: $BACKUP_DIR/backup_$TIMESTAMP"

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +

# File: scripts/run-tests.sh
#!/bin/bash

echo "🧪 Running Tests..."

# Backend Tests
echo "Testing Backend..."
cd backend
go test ./... -v
BACKEND_STATUS=$?
cd ..

# ML Service Tests
echo "Testing ML Service..."
cd ml-service
python -m pytest tests/ -v
ML_STATUS=$?
cd ..

# Frontend Tests
echo "Testing Frontend..."
cd frontend
npm test -- --watchAll=false
FRONTEND_STATUS=$?
cd ..

# Summary
echo ""
echo "Test Summary:"
echo "- Backend: $([ $BACKEND_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "- ML Service: $([ $ML_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "- Frontend: $([ $FRONTEND_STATUS -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"

# Exit with error if any test failed
[ $BACKEND_STATUS -eq 0 ] && [ $ML_STATUS -eq 0 ] && [ $FRONTEND_STATUS -eq 0 ]