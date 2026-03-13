# Prescription Analyzer AI: Journey & Solutions

This document serves as a comprehensive log of the technical challenges faced and the solutions implemented during the development and refinement of the Prescription Analyzer AI.

## 🔧 Issues Addressed & Solved

### 1. 0% Extraction — No medicines, No patient info (26% confidence)
*   **Cause**: Tesseract OCR running 20 passes (4 configs × 5 variants) was timing out.
*   **Fix**: Reduced to 2 smart OCR passes (PSM 6 + PSM 11).

### 2. 2-Minute Timeout (context deadline exceeded)
*   **Cause**: Go backend had `Timeout: 120s`, OCR was taking longer.
*   **Fix**: Increased Go HTTP client timeout to `300s`.

### 3. Doctor Name Extracted as "Dr. Unita MU" instead of "Sunita Mehta"
*   **Cause**: `Dr. (Mrs.)` parenthetical broke old regex pattern.
*   **Fix**: New regex handles optional bracketed titles like `(Mrs.)`.

### 4. Medicines recognized (Esofag, Bifilac, Emanzen)
*   **Cause**: These Indian brand names were missing from `MEDICINE_DB`.
*   **Fix**: Added 60+ medicines including gynaecology-specific drugs.

### 5. Handwriting not readable — OCR fundamentally broken for handwriting
*   **Cause**: Tesseract is a font-trained OCR, not designed for doctor handwriting.
*   **Fix**: Switched primary engine to Claude Vision → then Gemini Vision which reads handwriting like a human.

### 6. Anthropic API: "Credit balance too low" (400 error)
*   **Cause**: Anthropic account had no credits.
*   **Fix**: Replaced Anthropic with Gemini 2.0 Flash (free tier, 1500 req/day).

### 7. `google.generativeai` FutureWarning — deprecated SDK
*   **Cause**: Old SDK (`google-generativeai`) was deprecated by Google.
*   **Fix**: Migrated to new SDK (`google-genai`) with updated API calls:
    ```python
    # Old
    import google.generativeai as genai
    # New
    from google import genai
    ```

### 8. GEMINI_API_KEY not loading despite being set
*   **Cause 1**: Space before value in `.env` — `GEMINI_API_KEY= AIza...`
*   **Cause 2**: `importlib` reading cached module before key was added.
*   **Fix**: New `_load_key()` function with 4 fallback strategies — env → `.env` file parse → importlib with reload → raw file parse. Also `.strip()` on all values.

### 9. Gemini 429 Rate Limit — sleeping 60s then giving up
*   **Cause**: Single `time.sleep(60)` then falling back to Cohere permanently.
*   **Fix**: Exponential backoff retry: `5s → 15s → 30s` (3 attempts before fallback).

### 10. API Keys exposed on GitHub
*   **Cause**: `COHERE_API_KEY` and `GEMINI_API_KEY` hardcoded in `integration/keys.py`.
*   **Fix**:
    *   `keys.py` now reads from `.env` only — safe to commit.
    *   Added `.env` to `.gitignore`.
    *   Advised regenerating both exposed keys immediately.

### 11. Frequency showing short codes (`bd`, `SOS`) instead of full labels
*   **Cause**: `gemini_system_prompt.txt` had rules but no concrete WRONG/CORRECT examples — Gemini ignored them.
*   **Fix**: Added explicit examples in system prompt:
    ```
    WRONG:   "frequency": "bd"
    CORRECT: "frequency": "Twice daily (Morning & Night)"
    ```

### 12. Indian prescription shorthand not understood
*   **Cause**: No context about Indian-specific conventions.
*   **Fix**: Taught the system all shorthand:
    *   `bd` = Before Dinner = twice daily
    *   `× 2` = 2 weeks duration (circled number = weeks)
    *   `·· —` (dots + line) = twice daily
    *   `c/o` = complaints of → diagnosis
    *   `Adv.` = start of medicine list
    *   `SOS` = emergency use only

---

## 🔁 Architecture Refactor

**Problem**: ML service had a self-learning feedback loop — storing corrections, maintaining a knowledge base, retraining on user input.
**Fix**: Ripped out the entire learning pipeline. Replaced with a fully stateless scan-only flow: upload → OCR → extract → return result. Nothing is stored between requests.

## 🗑️ Dead Code & Files Cleanup

**Problem**: Old files from the self-learning era still sitting in the project — `continuous_learner.py`, `training_service.py`, `medicine_database.pkl`, `models/` folder, stale `__pycache__` directories.
**Fix**: Identified every file to delete and cleaned up the structure. The `.pkl` file was the stored knowledge base accumulating over time.

## 🔌 Wrong Run Command

**Problem**: Running `python -m app.main_ml` threw `Could not import module "app.main"` because uvicorn always looks for `app.main`.
**Fix**: Rename `main_ml.py` → `main.py` and always run via `uvicorn app.main:app`.

## 💻 CPU Overload / Laptop Hanging

**Problem**: A single prescription upload was spawning up to 12 OCR passes — 3 preprocessed image variants × EasyOCR + 3 Tesseract configs each. This pegged all cores to 800%+ CPU.
**Fixes applied**:
*   Reduced to 1 image variant instead of 3.
*   One OCR engine at a time (Tesseract by default, EasyOCR opt-in via env var).
*   One Tesseract config instead of 3.
*   Hard image size cap at 2000px longest edge.
*   Thread caps set at import time: `OMP_NUM_THREADS=1`, `OpenCV setNumThreads(1)`, and all BLAS/MKL equivalents.
*   Removed `fuzzywuzzy` fuzzy matching on every medicine line (replaced with plain dict lookup).
*   `EasyOCR workers=0` to prevent subprocess forking.

## 🤖 Cohere Model Removed (404 Error)

**Problem**: Code was calling `model="command-r"` which Cohere retired on September 15, 2025. Every LLM call was returning 404.
**Fix**: Updated to `model="command-r-plus-08-2024"`.

## 📷 Poor OCR Quality on Handwriting (26% confidence)

**Problem**: Tesseract was scoring only 0.24 confidence on a real handwritten prescription.
**Fixes**:
*   Image cap raised from 1200px to 2000px — preserves fine ink strokes.
*   Preprocessing pipeline improved: added denoising step + sharpening kernel before CLAHE + adaptive threshold.
*   Adaptive threshold block size increased (11→31).
*   PSM 11 fallback added — if PSM 6 scores below 0.40 confidence, automatically retries with sparse-text mode.

## 🎨 Frontend Cleanup

**Problem**: `App.jsx` had feedback/correction UI from the old self-learning system.
**Fix**: Rewrote `App.jsx` as a clean scan-only UI: upload zone → loading spinner → results panel. No correction UI, no feedback submission.

---

## 📊 Final Result Comparison

| Metric | Start | End |
| :--- | :--- | :--- |
| **Confidence** | 26% | 95-99% |
| **Medicines found** | 0 | 4/4 |
| **Doctor name** | Empty | Sunita Mehta |
| **Patient name** | N/A | Banani |
| **Diagnosis** | Empty | Pain Vagina, Cold |
| **Engine** | Tesseract only | Gemini 2.0 Flash Vision |
| **API cost** | Paid (Anthropic) | Free (Gemini) |

---
*Maintained by Sneha Das*
