# Prescription Analyzer AI

An AI-powered system designed to analyze and extract information from digital and handwritten prescriptions. The project combines advanced OCR techniques with Biomedical Named Entity Recognition (NER) to provide accurate medical data extraction.

## 🚀 Current Project Status

The project is currently in an advanced development phase with core functionality established across the microservices architecture.

### Key Achievements:
- **TrOCR Integration**: Successfully integrated Microsoft's Transformer-based OCR (TrOCR) to handle complex handwritten text in prescriptions.
- **Biomedical NER**: Implemented `d4data/biomedical-ner-all` for robust extraction of symptoms, medications, and dosages.
- **Hybrid Backend**: Transitioned to a Go-based backend for efficient API handling while maintaining a specialized Python ML service.
- **Data Persistence**: Integrated **MongoDB** with the Go driver for secure and scalable data storage.
- **Improved UI**: Added **Pagination** and other navigational enhancements for better data management in the frontend.
- **Microservices Architecture**: Separate services for Frontend (React), Backend (Go), and ML (Python) are communicating effectively.
- **Cloud Deployment Ready**: Configuration complete for Railway (Backend/ML) and Vercel (Frontend).

## 🎯 Next Focus Areas

1.  **Confidence Rate Optimization**: Improving the model's confidence scores for highly illegible handwriting.
2.  **Dataset Expansion**: Further training/fine-tuning TrOCR on a broader set of medical prescription imagery.
3.  **Advanced UI/UX**: Enhancing the frontend to handle and visualize complex analysis results (e.g., medicine interactions, dosage warnings).
4.  **Performance Tuning**: Optimizing the latency between the Go backend and Python ML service.

## 🏗️ Tech Stack

- **Frontend**: React 18
- **Backend**: Go (Golang)
- **ML Service**: Python (FastAPI/HuggingFace Transformers)
- **Models**: TrOCR, d4data/biomedical-ner-all
- **Deployment**: Railway, Vercel
- **Database**: MongoDB (Go Driver integrated)

---
*For a detailed history of challenges and how they were overcome, see [ISSUES_RESOLVED.md](./ISSUES_RESOLVED.md).*