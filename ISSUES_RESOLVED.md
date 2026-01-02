# Project Development: Issues & Solutions

This document tracks the major challenges encountered during the development of the Prescription Analyzer AI and the solutions implemented to resolve them.

## 🛠️ Resolved Issues

### 1. Deployment Complexity (Railway/Vercel)
- **Issue**: Railway had trouble detecting the correct primary language (Go vs Python) for the backend service.
- **Solution**: 
  - Forced Python detection where necessary using `runtime.txt`.
  - Created `start.sh` to explicitly manage the execution environment.
  - Configured `Procfile` for Railway deployments.
  - Downgraded React to version 18 for compatibility with existing Create React App (CRA) configurations.

### 2. OCR & Model Performance
- **Issue**: Standard OCR (Tesseract) struggled with handwriting, and initial NER models were private or inaccessible.
- **Solution**: 
  - Replaced private `clinical-ai/Clinical-Bert-NER` with the open-source and effective `d4data/biomedical-ner-all`.
  - Integrated **TrOCR** (Transformer-based OCR) for significantly better handwriting recognition.
  - Implemented confidence check logic to flag uncertain predictions.

### 3. Backend-ML Communication Sync
- **Issue**: Discrepancies between the Go backend responses and the Python ML service outputs during early integration.
- **Solution**: 
  - Developed `debug.py` and other debugging tools to test both services in isolation and verify data consistency.
  - Standardized JSON schemas across both services to ensure data alignment.

### 4. API & Infrastructure Errors
- **Issue**: Encountered persistent HTTPS 500 errors and missing file path errors during file uploads.
- **Solution**: 
  - Refined error handling in `prescription_handler.go`.
  - Standardized data paths (e.g., transitioning from `data/prescriptions/` to relative paths like `../data/prescriptions`).
  - Implemented `.env` management to prevent sensitive data leaks while ensuring local development consistency.

### 5. UI/UX Data Display
- **Issue**: Prescription analysis would succeed on the backend, but the output wouldn't render correctly on the frontend.
- **Solution**: 
  - Debugged the frontend state management to ensure it correctly parses the enhanced JSON response from the hybrid backend.

## 📈 Evolution Timeline

| Phase | Milestone | Focus |
| :--- | :--- | :--- |
| **Early Dec** | Foundation | Setting up Go/Python hybrid architecture. |
| **Mid Dec** | Integration | Moving to TrOCR and fixing service communication. |
| **Late Dec** | Optimization | Training model on specific datasets and improving confidence. |
| **Early Jan** | Deployment | Solving cloud environment conflicts and finalizing CI/CD. |

---
*Created by Snehadas2005*
