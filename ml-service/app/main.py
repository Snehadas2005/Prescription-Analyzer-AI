import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.extraction_service import ExtractionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prescription Analyzer — ML Service",
    description="Stateless OCR + LLM prescription extraction. No data stored.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extraction_service: ExtractionService | None = None


@app.on_event("startup")
async def startup_event():
    global extraction_service
    logger.info("🚀 Starting ML Service (stateless mode)…")
    extraction_service = ExtractionService()
    logger.info("✅ ML Service ready")


@app.get("/")
async def root():
    return {
        "service": "Prescription Analyzer — ML Service",
        "mode": "stateless",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-prescription",
            "extract": "/extract",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "analyzer_ready": extraction_service is not None,
        "version": "2.0.0",
    }


@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    """Original endpoint — kept for backward compatibility."""
    return await _analyse(file)


@app.post("/analyze-prescription")
async def analyze_prescription(file: UploadFile = File(...)):
    """Primary endpoint called by the Go backend."""
    return await _analyse(file)


async def _analyse(file: UploadFile) -> JSONResponse:
    if not extraction_service:
        raise HTTPException(503, "Service not ready — please retry in a moment.")

    # Accept any image content-type; also validate by extension as fallback
    valid_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
    filename = (file.filename or "").lower()
    has_valid_ext = any(filename.endswith(ext) for ext in valid_extensions)
    is_image_ct = file.content_type and file.content_type.startswith("image/")

    if not is_image_ct and not has_valid_ext:
        raise HTTPException(
            400,
            f"Invalid file. Expected an image (JPEG, PNG, etc.), got: {file.content_type}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large — maximum 10 MB.")
    if len(image_bytes) < 100:
        raise HTTPException(400, "File too small — please upload a valid image.")

    if not _is_image(image_bytes):
        raise HTTPException(400, "File does not appear to be a valid image.")

    try:
        result = await extraction_service.extract(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}")


def _is_image(data: bytes) -> bool:
    """Quick magic-byte check."""
    sigs = [
        b"\xFF\xD8\xFF",       # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a", b"GIF89a",  # GIF
        b"BM",                  # BMP
        b"II\x2A\x00",          # TIFF LE
        b"MM\x00\x2A",          # TIFF BE
        b"RIFF",                # WebP
    ]
    for sig in sigs:
        if data.startswith(sig):
            return True
    if data.startswith(b"RIFF") and b"WEBP" in data[:20]:
        return True
    return False


if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting on {HOST}:{PORT}")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, log_level="info")