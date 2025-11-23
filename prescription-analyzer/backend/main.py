from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import tempfile
import uuid
import logging
from datetime import datetime, timedelta
import asyncio

# Import our enhanced prescription analyzer
try:
    from prescription_analyzer import EnhancedPrescriptionAnalyzer
except ImportError as e:
    print(f"Failed to import EnhancedPrescriptionAnalyzer: {e}")
    print("Please ensure prescription_analyzer.py is in the same directory")
    raise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Prescription Analyzer API",
    description="Advanced prescription analysis using OCR and NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: Optional[EnhancedPrescriptionAnalyzer] = None

# Pydantic models for request/response
class OrderRequest(BaseModel):
    prescription_id: str
    patient_info: Dict[str, Any]
    medicines: List[Dict[str, Any]]
    delivery_address: str
    contact_number: str

class OrderResponse(BaseModel):
    success: bool
    order_id: str = ""
    message: str = ""
    estimated_delivery: str = ""
    total_amount: float = 0.0
    error: str = ""

# Fixed AnalysisResponse with proper null handling
class AnalysisResponse(BaseModel):
    success: bool = Field(default=False)
    prescription_id: str = Field(default="")
    patient: Dict[str, str] = Field(default_factory=dict)
    doctor: Dict[str, str] = Field(default_factory=dict) 
    medicines: List[Dict[str, Any]] = Field(default_factory=list)
    diagnosis: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0)
    patient_name: str = Field(default="")
    patient_age: int = Field(default=0)
    patient_gender: str = Field(default="")
    doctor_name: str = Field(default="")
    doctor_license: str = Field(default="")
    message: str = Field(default="")
    error: str = Field(default="")

    class Config:
        # Allow extra fields and handle validation more gracefully
        extra = "ignore"

class HealthResponse(BaseModel):
    status: str
    analyzer_ready: bool
    timestamp: str
    uptime: str = "N/A"
    cohere_available: bool = False

# In-memory storage for demo purposes (use database in production)
orders_storage: Dict[str, Dict] = {}
prescriptions_storage: Dict[str, Any] = {}

def ensure_safe_response_data(data: dict) -> dict:
    """
    Ensure all response data is safe for Pydantic validation
    Converts None values to appropriate defaults
    """
    safe_data = {}
    
    # Handle all possible fields with safe defaults
    safe_data['success'] = data.get('success', False)
    safe_data['prescription_id'] = data.get('prescription_id') or ""
    safe_data['patient'] = data.get('patient') or {}
    safe_data['doctor'] = data.get('doctor') or {}
    safe_data['medicines'] = data.get('medicines') or []
    safe_data['diagnosis'] = data.get('diagnosis') or []
    safe_data['confidence_score'] = data.get('confidence_score', 0.0)
    safe_data['message'] = data.get('message') or ""
    safe_data['error'] = data.get('error') or ""
    
    # Legacy fields with null safety
    safe_data['patient_name'] = data.get('patient_name') or ""
    safe_data['patient_gender'] = data.get('patient_gender') or ""
    safe_data['doctor_name'] = data.get('doctor_name') or ""
    safe_data['doctor_license'] = data.get('doctor_license') or ""
    
    # Handle patient_age specially (convert to int)
    patient_age = data.get('patient_age', 0)
    try:
        safe_data['patient_age'] = int(patient_age) if patient_age is not None else 0
    except (ValueError, TypeError):
        safe_data['patient_age'] = 0
    
    # Ensure nested dictionaries are also safe
    if isinstance(safe_data['patient'], dict):
        patient_dict = safe_data['patient']
        patient_dict['name'] = patient_dict.get('name') or ""
        patient_dict['age'] = patient_dict.get('age') or ""
        patient_dict['gender'] = patient_dict.get('gender') or ""
    
    if isinstance(safe_data['doctor'], dict):
        doctor_dict = safe_data['doctor']
        doctor_dict['name'] = doctor_dict.get('name') or ""
        doctor_dict['specialization'] = doctor_dict.get('specialization') or ""
        doctor_dict['registration_number'] = doctor_dict.get('registration_number') or ""
    
    # Ensure medicines list is safe
    if isinstance(safe_data['medicines'], list):
        safe_medicines = []
        for med in safe_data['medicines']:
            if isinstance(med, dict):
                safe_med = {
                    'name': med.get('name') or "",
                    'dosage': med.get('dosage') or "",
                    'quantity': med.get('quantity') or "",
                    'frequency': med.get('frequency') or "",
                    'duration': med.get('duration') or "",
                    'instructions': med.get('instructions') or "",
                    'available': med.get('available', True)
                }
                safe_medicines.append(safe_med)
        safe_data['medicines'] = safe_medicines
    
    return safe_data

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup"""
    global analyzer
    try:
        logger.info("Initializing Enhanced Prescription Analyzer...")
        
        # Get configuration from environment variables
        cohere_api_key = os.getenv('COHERE_API_KEY')
        tesseract_path = os.getenv('TESSERACT_PATH')
        
        # Initialize analyzer with forced API usage
        analyzer = EnhancedPrescriptionAnalyzer(
            cohere_api_key=cohere_api_key,
            tesseract_path=tesseract_path,
            force_api=True  # Force API usage
        )
        
        logger.info("Enhanced Prescription Analyzer initialized successfully")
        
        # Log configuration status
        if hasattr(analyzer, 'co') and analyzer.co:
            logger.info("Cohere API key found - Advanced NLP analysis available")
        else:
            logger.warning("No Cohere API key - Using pattern-based analysis fallback")
            
        if tesseract_path:
            logger.info(f"Tesseract path configured: {tesseract_path}")
        else:
            logger.info("Using system default Tesseract installation")
            
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        raise RuntimeError(f"Analyzer initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Prescription Analyzer API")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Prescription Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze": "/api/analyze-prescription",
            "create_order": "/api/create-order",
            "medicines": "/api/medicines/search"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    cohere_available = analyzer is not None and hasattr(analyzer, 'co') and analyzer.co is not None
    
    return HealthResponse(
        status="healthy" if analyzer is not None else "unhealthy",
        analyzer_ready=analyzer is not None,
        timestamp=datetime.now().isoformat(),
        uptime="N/A",  # Would implement proper uptime tracking in production
        cohere_available=cohere_available
    )

@app.post("/api/analyze-prescription", response_model=AnalysisResponse)
async def analyze_prescription(file: UploadFile = File(...)):
    """
    Analyze uploaded prescription image
    
    Args:
        file: Uploaded image file (JPEG, PNG, TIFF)
        
    Returns:
        AnalysisResponse with extracted prescription data
        
    Raises:
        HTTPException: If analysis fails or file is invalid
    """
    if not analyzer:
        raise HTTPException(
            status_code=503, 
            detail="Analyzer not initialized. Please check server configuration."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file (JPEG, PNG, TIFF, BMP, WEBP)"
        )
    
    # Check file size (max 10MB)
    temp_file_path = None
    
    try:
        # Create temporary file with proper extension
        file_extension = '.jpg'  # Default
        if file.content_type:
            if 'png' in file.content_type:
                file_extension = '.png'
            elif 'tiff' in file.content_type:
                file_extension = '.tiff'
            elif 'bmp' in file.content_type:
                file_extension = '.bmp'
            elif 'webp' in file.content_type:
                file_extension = '.webp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            
            # Read and save file
            content = await file.read()
            file_size = len(content)
            
            # Check file size limit (10MB)
            if file_size > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=413, 
                    detail="File too large. Maximum size is 10MB"
                )
            
            if file_size < 1024:  # Less than 1KB
                raise HTTPException(
                    status_code=400,
                    detail="File too small. Please ensure the image is readable"
                )
            
            temp_file.write(content)
            temp_file.flush()
        
        logger.info(f"Processing prescription file: {file.filename}, Size: {file_size} bytes")
        
        # Analyze prescription using the enhanced analyzer
        result = analyzer.analyze_prescription(temp_file_path)
        
        if not result.success:
            logger.warning(f"Analysis failed for prescription {result.prescription_id}: {result.error}")
            # Return safe failure response
            safe_error_data = ensure_safe_response_data({
                'success': False,
                'error': result.error,
                'message': "Failed to analyze prescription. Please ensure the image is clear and contains readable text."
            })
            return AnalysisResponse(**safe_error_data)
        
        # Store result for potential order creation
        prescriptions_storage[result.prescription_id] = result
        
        # Convert to JSON format expected by frontend
        json_result = analyzer.to_json(result)
        
        logger.info(f"Analysis completed successfully for prescription {result.prescription_id} with confidence {result.confidence_score:.2f}")
        
        # Ensure response data is safe for Pydantic validation
        safe_data = ensure_safe_response_data(json_result)
        
        # Return structured response
        try:
            return AnalysisResponse(**safe_data)
        except Exception as validation_error:
            logger.error(f"Validation error creating response: {validation_error}")
            logger.error(f"Response data: {safe_data}")
            
            # Fall back to minimal safe response
            fallback_data = ensure_safe_response_data({
                'success': result.success,
                'prescription_id': result.prescription_id,
                'message': 'Analysis completed but response formatting had issues',
                'confidence_score': result.confidence_score
            })
            return AnalysisResponse(**fallback_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing prescription: {str(e)}")
        safe_error_data = ensure_safe_response_data({
            'success': False,
            'error': str(e),
            'message': "An unexpected error occurred during analysis. Please try again."
        })
        return AnalysisResponse(**safe_error_data)
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

@app.post("/api/create-order", response_model=OrderResponse)
async def create_order(order_request: OrderRequest):
    """
    Create medicine order from analyzed prescription
    
    Args:
        order_request: Order details including prescription ID, medicines, and delivery info
        
    Returns:
        OrderResponse with order confirmation details
        
    Raises:
        HTTPException: If order creation fails
    """
    try:
        # Validate prescription exists
        if order_request.prescription_id not in prescriptions_storage:
            raise HTTPException(
                status_code=404, 
                detail="Prescription not found. Please analyze prescription first."
            )
        
        # Validate required fields
        if not order_request.delivery_address or not order_request.delivery_address.strip():
            raise HTTPException(status_code=400, detail="Delivery address is required")
        
        if not order_request.contact_number or not order_request.contact_number.strip():
            raise HTTPException(status_code=400, detail="Contact number is required")
        
        if not order_request.medicines or len(order_request.medicines) == 0:
            raise HTTPException(status_code=400, detail="No medicines selected for order")
        
        # Validate medicines
        for medicine in order_request.medicines:
            if not medicine.get('name') or not medicine.get('name').strip():
                raise HTTPException(status_code=400, detail="All medicines must have valid names")
            
            quantity = medicine.get('quantity', 1)
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                raise HTTPException(status_code=400, detail="All medicines must have valid quantities")
        
        # Generate order ID
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate estimated delivery (2-3 days from now)
        estimated_delivery = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        
        # Calculate total amount (mock calculation based on medicines)
        base_price_per_medicine = 75.0  # ₹75 per medicine (demo)
        delivery_charge = 50.0  # ₹50 delivery charge
        total_amount = delivery_charge
        
        for medicine in order_request.medicines:
            quantity = float(medicine.get('quantity', 1))
            medicine_cost = base_price_per_medicine * quantity
            
            # Apply discount for bulk orders
            if quantity >= 5:
                medicine_cost *= 0.9  # 10% discount
            elif quantity >= 3:
                medicine_cost *= 0.95  # 5% discount
            
            total_amount += medicine_cost
        
        # Round to 2 decimal places
        total_amount = round(total_amount, 2)
        
        # Store order
        order_data = {
            "order_id": order_id,
            "prescription_id": order_request.prescription_id,
            "patient_info": order_request.patient_info,
            "medicines": order_request.medicines,
            "delivery_address": order_request.delivery_address,
            "contact_number": order_request.contact_number,
            "status": "confirmed",
            "created_at": datetime.now().isoformat(),
            "estimated_delivery": estimated_delivery,
            "total_amount": total_amount,
            "delivery_charge": delivery_charge
        }
        
        orders_storage[order_id] = order_data
        
        logger.info(f"Order created successfully: {order_id} for prescription {order_request.prescription_id}")
        
        return OrderResponse(
            success=True,
            order_id=order_id,
            message=f"Order placed successfully! Your order ID is {order_id}",
            estimated_delivery=estimated_delivery,
            total_amount=total_amount
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        return OrderResponse(
            success=False,
            error=str(e),
            message="Failed to create order. Please check your details and try again."
        )

@app.get("/api/order/{order_id}")
async def get_order(order_id: str):
    """
    Get order details by ID
    
    Args:
        order_id: The order ID to retrieve
        
    Returns:
        Order details dictionary
        
    Raises:
        HTTPException: If order not found
    """
    if order_id not in orders_storage:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order_data = orders_storage[order_id]
    logger.info(f"Retrieved order details for {order_id}")
    
    return {
        "success": True,
        "order": order_data
    }

@app.get("/api/orders")
async def list_orders(limit: int = 50, offset: int = 0):
    """
    List all orders with pagination
    
    Args:
        limit: Maximum number of orders to return (default: 50)
        offset: Number of orders to skip (default: 0)
        
    Returns:
        Dictionary with orders list and pagination info
    """
    all_orders = list(orders_storage.values())
    
    # Sort by creation date (newest first)
    all_orders.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Apply pagination
    paginated_orders = all_orders[offset:offset + limit]
    
    return {
        "success": True,
        "orders": paginated_orders,
        "total_count": len(all_orders),
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < len(all_orders)
    }

@app.get("/api/prescription/{prescription_id}")
async def get_prescription(prescription_id: str):
    """
    Get prescription analysis by ID
    
    Args:
        prescription_id: The prescription ID to retrieve
        
    Returns:
        Prescription analysis data
        
    Raises:
        HTTPException: If prescription not found
    """
    if prescription_id not in prescriptions_storage:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    result = prescriptions_storage[prescription_id]
    logger.info(f"Retrieved prescription details for {prescription_id}")
    
    return {
        "success": True,
        "prescription": analyzer.to_json(result)
    }

@app.delete("/api/prescription/{prescription_id}")
async def delete_prescription(prescription_id: str):
    """
    Delete prescription data
    
    Args:
        prescription_id: The prescription ID to delete
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If prescription not found
    """
    if prescription_id not in prescriptions_storage:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    del prescriptions_storage[prescription_id]
    logger.info(f"Deleted prescription {prescription_id}")
    
    return {
        "success": True,
        "message": "Prescription deleted successfully"
    }

@app.get("/api/medicines/search")
async def search_medicines(query: str = "", limit: int = 20):
    """
    Search available medicines
    
    Args:
        query: Search query string (optional)
        limit: Maximum number of results to return (default: 20)
        
    Returns:
        Dictionary with matching medicines
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        if not query.strip():
            # Return first 20 medicines if no query
            medicines = list(analyzer.medicine_database.keys())[:limit]
            return {
                "success": True,
                "medicines": medicines,
                "total_found": len(medicines),
                "query": ""
            }
        
        # Simple search in medicine database
        query_lower = query.lower().strip()
        matches = []
        
        for name, info in analyzer.medicine_database.items():
            # Search in name and generic name
            if (query_lower in name.lower() or 
                query_lower in info.get('generic', '').lower() or
                query_lower in info.get('category', '').lower()):
                matches.append({
                    "name": name.title(),
                    "generic": info.get('generic', ''),
                    "category": info.get('category', ''),
                    "available": info.get('available', True)
                })
        
        # Sort by relevance (exact matches first)
        matches.sort(key=lambda x: (
            0 if query_lower == x['name'].lower() else
            1 if query_lower in x['name'].lower() else 2
        ))
        
        return {
            "success": True,
            "medicines": matches[:limit],
            "total_found": len(matches),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error searching medicines: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/api/medicines/{medicine_name}")
async def get_medicine_info(medicine_name: str):
    """
    Get detailed medicine information
    
    Args:
        medicine_name: Name of the medicine to look up
        
    Returns:
        Detailed medicine information
        
    Raises:
        HTTPException: If medicine not found
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        medicine_lower = medicine_name.lower().strip()
        
        # Direct match
        if medicine_lower in analyzer.medicine_database:
            info = analyzer.medicine_database[medicine_lower]
            return {
                "success": True,
                "medicine": {
                    "name": medicine_name.title(),
                    **info
                }
            }
        
        # Fuzzy search
        from fuzzywuzzy import process
        best_match = process.extractOne(
            medicine_lower, 
            analyzer.medicine_database.keys(),
            score_cutoff=70
        )
        
        if best_match:
            matched_name = best_match[0]
            info = analyzer.medicine_database[matched_name]
            return {
                "success": True,
                "medicine": {
                    "name": matched_name.title(),
                    "match_score": best_match[1],
                    **info
                },
                "message": f"Found close match: {matched_name.title()}"
            }
        
        raise HTTPException(status_code=404, detail="Medicine not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting medicine info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve medicine information")

@app.get("/api/stats")
async def get_stats():
    """
    Get API usage statistics
    
    Returns:
        Dictionary with current statistics
    """
    return {
        "success": True,
        "statistics": {
            "total_prescriptions_analyzed": len(prescriptions_storage),
            "total_orders_created": len(orders_storage),
            "medicines_in_database": len(analyzer.medicine_database) if analyzer else 0,
            "analyzer_status": "ready" if analyzer else "not_initialized",
            "cohere_available": (hasattr(analyzer, 'co') and analyzer.co is not None) if analyzer else False,
            "timestamp": datetime.now().isoformat()
        }
    }

# Global exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception in {request.url.path}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    logger.warning(f"Value error in {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Invalid input",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    RELOAD = os.getenv('RELOAD', 'true').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').lower()
    
    logger.info(f"Starting AI Prescription Analyzer API on {HOST}:{PORT}")
    logger.info(f"Reload mode: {RELOAD}")
    logger.info(f"Log level: {LOG_LEVEL}")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL,
        access_log=True
    )