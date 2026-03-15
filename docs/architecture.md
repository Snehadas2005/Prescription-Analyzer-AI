# System Architecture

The Prescription Analyzer AI is built on a high-performance, microservices-oriented architecture designed to combine the speed of Go with the machine learning capabilities of Python.

## 🏗️ Architecture Overview

The system is divided into three primary layers:

### 1. Frontend (React 18)
- **Role**: User Interface and Interaction.
- **Key Features**: 
  - Image upload with preview functionality.
  - Real-time display of analysis results.
  - Pagination for managing historical prescription data.
- **Deployment**: Vercel.

### 2. Backend (Go/Golang)
- **Role**: API Gateway, Data Orchestration, and Persistence.
- **Key Features**:
  - High-concurrency request handling.
  - Integration with **MongoDB** for secure data storage.
  - Proxying intensive ML tasks to the specialized ML Service.
  - File management (uploads/storage).
- **Deployment**: Railway.

### 3. ML Service (Python/FastAPI)
- **Role**: Heavy-duty Medical Intelligence.
- **Key Models**:
  - **Gemini 2.0 Flash Vision (Primary)**: State-of-the-art vision model for handwriting recognition and entity extraction across full prescription images.
  - **Cohere Command-R Plus (Secondary)**: Fallback model.
- **Deployment**: Railway.

## 🔄 Data Flow

1.  **Ingestion**: User uploads a prescription image via the React frontend.
2.  **Orchestration**: The Go backend receives the file, stores it, and sends a processing request to the ML service.
3.  **Analysis**:
    - The Python service applies preprocessing (e.g., contrast enhancement for regional scripts like Hindi).
    - The processed image is sent directly to Gemini 2.0 Flash Vision to concurrently extract text and clinical entities like symptoms, medicines, and dosages.
    - Confidence scores and translated data are integrated into the output.
4.  **Persistence**: Results are returned to the Go backend and persisted in MongoDB.
5.  **Visualization**: The backend responds to the frontend with a structured JSON, which is then rendered for the user.

## 🛡️ Security & Reliability
- **Isolated ML Environment**: Machine learning dependencies are isolated from the main API logic to prevent resource contention.
- **Schema Validation**: Standardized data transfer objects (DTOs) ensure consistent communication between Go and Python.
- **Environment Management**: Strict separation of development and production configurations via `.env` files.
