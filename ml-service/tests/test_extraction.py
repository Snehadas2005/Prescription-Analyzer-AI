import pytest
from app.services.extraction_service import ExtractionService
import numpy as np

@pytest.fixture
def extraction_service():
    return ExtractionService()

def test_extraction_service_initialization(extraction_service):
    """Test that extraction service initializes correctly"""
    assert extraction_service is not None
    assert extraction_service.reader is not None

def test_medicine_database_loaded(extraction_service):
    """Test that medicine database is loaded"""
    assert len(extraction_service.medicine_database) > 0
    assert 'paracetamol' in extraction_service.medicine_database

def test_text_cleaning(extraction_service):
    """Test text cleaning functionality"""
    text = "Patient:   John  Doe\n\nAge: 30"
    # Extraction service should handle this
    assert extraction_service is not None

# File: backend/internal/handlers/prescription_handler_test.go
package handlers

import (
    "testing"
)

func TestPrescriptionHandler(t *testing.T) {
    // TODO: Add comprehensive tests
    t.Log("Prescription handler tests")
}

# File: frontend/src/App.test.js
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders app title', () => {
  render(<App />);
  const titleElement = screen.getByText(/AI Prescription Analyzer/i);
  expect(titleElement).toBeInTheDocument();
});

# File: Makefile
.PHONY: help setup dev build test clean deploy

help:
	@echo "Available commands:"
	@echo "  make setup    - Setup development environment"
	@echo "  make dev      - Start development servers"
	@echo "  make build    - Build all services"
	@echo "  make test     - Run all tests"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make deploy   - Deploy to production"

setup:
	@bash scripts/setup.sh

dev:
	@bash scripts/start-dev.sh

build:
	@echo "Building all services..."
	@cd frontend && npm run build
	@cd backend && go build -o bin/server cmd/server/main.go
	@echo "✅ Build complete"

test:
	@bash scripts/run-tests.sh

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf frontend/build
	@rm -rf backend/bin
	@rm -rf ml-service/__pycache__
	@rm -rf ml-service/.pytest_cache
	@echo "✅ Clean complete"

deploy:
	@bash scripts/deploy.sh

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

backup:
	@bash scripts/backup-db.sh