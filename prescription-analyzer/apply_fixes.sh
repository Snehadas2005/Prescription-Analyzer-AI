#!/bin/bash

echo "🔧 Applying fixes..."

# Backup original files
cp backend/internal/handlers/prescription_handler.go backend/internal/handlers/prescription_handler.go.backup 2>/dev/null || true
cp backend/main.py backend/main.py.backup 2>/dev/null || true

# Apply fixes
cp backend/internal/handlers/prescription_handler_fixed.go backend/internal/handlers/prescription_handler.go
cp backend/main_fixed.py backend/main.py

echo "✅ Fixes applied!"
echo ""
echo "Now restart your services:"
echo "  logs/startup.sh"
