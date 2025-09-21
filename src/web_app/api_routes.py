from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import uuid
from datetime import datetime
import json

# Import OMR processing modules
from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_extractor import AnswerExtractor
from omr_processor.sheet_classifier import SheetClassifier
from utils.database import DatabaseManager
from utils.helpers import validate_image_file, setup_logging

# Pydantic models for request/response
class ProcessingRequest(BaseModel):
    sheet_version: Optional[str] = "AUTO_DETECT"
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    save_to_db: bool = True

class ProcessingResponse(BaseModel):
    success: bool
    processing_id: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StudentInfo(BaseModel):
    id: str
    name: str
    email: Optional[str] = None

class AnswerKeyRequest(BaseModel):
    set_version: str
    answer_key: Dict[int, str]

# Initialize FastAPI app
app = FastAPI(
    title="OMR Evaluation API",
    description="REST API for automated OMR sheet evaluation and scoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = ImagePreprocessor()
bubble_detector = BubbleDetector()
answer_extractor = AnswerExtractor()
sheet_classifier = SheetClassifier()
db_manager = DatabaseManager()
logger = setup_logging()

# Processing job storage (in production, use Redis or similar)
processing_jobs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting OMR Evaluation API")
    
    # Ensure required directories exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OMR Evaluation API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "process": "/api/v1/process",
            "results": "/api/v1/results",
            "answer_keys": "/api/v1/answer-keys"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        stats = db_manager.get_statistics()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "total_results": stats['total_results']
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/v1/process", response_model=ProcessingResponse)
async def process_omr_sheet(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sheet_version: str = "AUTO_DETECT",
    student_id: Optional[str] = None,
    student_name: Optional[str] = None,
    save_to_db: bool = True
):
    """
    Process a single OMR sheet image.
    
    Args:
        file: Uploaded image file
        sheet_version: Sheet version or AUTO_DETECT
        student_id: Optional student ID
        student_name: Optional student name
        save_to_db: Whether to save results to database
    
    Returns:
        Processing response with results
    """
    # Generate processing ID
    processing_id = str(uuid.uuid4())
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Validate image file
        validation_result = validate_image_file(temp_path)
        if not validation_result['is_valid']:
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image file: {validation_result['error_message']}"
            )
        
        # Process asynchronously
        background_tasks.add_task(
            process_image_background,
            processing_id,
            temp_path,
            file.filename,
            sheet_version,
            student_id,
            student_name,
            save_to_db
        )
        
        # Store initial job status
        processing_jobs[processing_id] = {
            "status": "processing",
            "created_at": datetime.now(),
            "filename": file.filename,
            "student_id": student_id,
            "student_name": student_name
        }
        
        return ProcessingResponse(
            success=True,
            processing_id=processing_id,
            message="Processing started successfully"
        )
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Error processing OMR sheet: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_background(
    processing_id: str,
    image_path: str,
    filename: str,
    sheet_version: str,
    student_id: Optional[str],
    student_name: Optional[str],
    save_to_db: bool
):
    """
    Background task to process OMR image.
    """
    try:
        start_time = datetime.now()
        
        # Step 1: Preprocess image
        preprocessed_image = preprocessor.preprocess_image(image_path)
        
        # Step 2: Classify sheet
        if sheet_version == 'AUTO_DETECT':
            classification_result = sheet_classifier.classify_sheet(preprocessed_image)
            detected_version = classification_result.get('detected_set', 'SET_A')
        else:
            detected_version = sheet_version
            classification_result = {'detected_set': sheet_version, 'confidence': 1.0}
        
        # Step 3: Detect bubbles
        bubble_grid = bubble_detector.detect_answer_bubbles_grid(preprocessed_image)
        
        if not bubble_grid['success']:
            raise Exception('Failed to detect bubble grid')
        
        # Step 4: Extract answers
        answer_key = sheet_classifier.get_answer_key_for_set(detected_version)
        extraction_result = answer_extractor.extract_answers(
            preprocessed_image, bubble_grid, answer_key
        )
        
        if not extraction_result['success']:
            raise Exception('Failed to extract answers')
        
        # Step 5: Generate detailed report
        student_info = {
            'id': student_id or filename.split('.')[0],
            'name': student_name or filename.split('.')[0].replace('_', ' ').title(),
            'filename': filename
        }
        
        detailed_report = answer_extractor.generate_detailed_report(
            extraction_result, student_info
        )
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create final result
        result = {
            'success': True,
            'processing_id': processing_id,
            'timestamp': start_time,
            'filename': filename,
            'image_filename': filename,
            'sheet_version': detected_version,
            'student_info': student_info,
            'classification_result': classification_result,
            'bubble_detection_result': bubble_grid,
            'extraction_result': extraction_result,
            'detailed_report': detailed_report,
            'processing_time_seconds': processing_time,
            'status': 'completed'
        }
        
        # Add score summary
        if detailed_report.get('overall_score'):
            result.update({
                'score_summary': detailed_report['overall_score'],
                'subject_breakdown': detailed_report.get('subject_breakdown', [])
            })
        
        # Save to database if requested
        if save_to_db:
            db_manager.save_processing_result(result)
        
        # Update job status
        processing_jobs[processing_id] = {
            "status": "completed",
            "created_at": processing_jobs[processing_id]["created_at"],
            "completed_at": datetime.now(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        processing_jobs[processing_id] = {
            "status": "failed",
            "created_at": processing_jobs[processing_id]["created_at"],
            "failed_at": datetime.now(),
            "error": str(e)
        }
    
    finally:
        # Clean up temporary file
        if os.path.exists(image_path):
            os.unlink(image_path)

@app.get("/api/v1/process/{processing_id}")
async def get_processing_status(processing_id: str):
    """
    Get status of a processing job.
    
    Args:
        processing_id: Processing job ID
    
    Returns:
        Processing status and results
    """
    if processing_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = processing_jobs[processing_id]
    
    response = {
        "processing_id": processing_id,
        "status": job["status"],
        "created_at": job["created_at"].isoformat()
    }
    
    if job["status"] == "completed":
        response["completed_at"] = job["completed_at"].isoformat()
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["failed_at"] = job["failed_at"].isoformat()
        response["error"] = job["error"]
    
    return response

@app.post("/api/v1/process/batch")
async def process_batch_omr_sheets(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sheet_version: str = "AUTO_DETECT",
    save_to_db: bool = True
):
    """
    Process multiple OMR sheet images in batch.
    
    Args:
        files: List of uploaded image files
        sheet_version: Sheet version or AUTO_DETECT
        save_to_db: Whether to save results to database
    
    Returns:
        Batch processing response with job IDs
    """
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 50 files")
    
    batch_id = str(uuid.uuid4())
    processing_ids = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue  # Skip non-image files
        
        processing_id = str(uuid.uuid4())
        processing_ids.append(processing_id)
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Add background task
        background_tasks.add_task(
            process_image_background,
            processing_id,
            temp_path,
            file.filename,
            sheet_version,
            None,  # student_id
            None,  # student_name
            save_to_db
        )
        
        # Initialize job status
        processing_jobs[processing_id] = {
            "status": "processing",
            "created_at": datetime.now(),
            "filename": file.filename,
            "batch_id": batch_id
        }
    
    return {
        "success": True,
        "batch_id": batch_id,
        "processing_ids": processing_ids,
        "message": f"Batch processing started for {len(processing_ids)} files"
    }

@app.get("/api/v1/results")
async def get_results(
    limit: int = 20,
    offset: int = 0,
    student_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """
    Get processing results with optional filtering.
    
    Args:
        limit: Maximum number of results
        offset: Number of results to skip
        student_id: Filter by student ID
        date_from: Filter by start date (YYYY-MM-DD)
        date_to: Filter by end date (YYYY-MM-DD)
    
    Returns:
        List of processing results
    """
    try:
        # Convert date strings to datetime objects
        date_from_obj = None
        date_to_obj = None
        
        if date_from:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
        if date_to:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
        
        results = db_manager.get_processing_results(
            limit=limit,
            offset=offset,
            student_id=student_id,
            date_from=date_from_obj,
            date_to=date_to_obj
        )
        
        return {
            "success": True,
            "total": len(results),
            "limit": limit,
            "offset": offset,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/results/{result_id}")
async def get_result_by_id(result_id: int):
    """
    Get a specific processing result by ID.
    
    Args:
        result_id: Result ID
    
    Returns:
        Processing result details
    """
    result = db_manager.get_processing_result(result_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {
        "success": True,
        "result": result
    }

@app.get("/api/v1/statistics")
async def get_statistics():
    """
    Get system statistics.
    
    Returns:
        System statistics including totals, averages, and trends
    """
    try:
        stats = db_manager.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/answer-keys", response_model=dict)
async def save_answer_key(request: AnswerKeyRequest):
    """
    Save an answer key for a specific sheet version.
    
    Args:
        request: Answer key request containing set version and answers
    
    Returns:
        Success response
    """
    try:
        success = db_manager.save_answer_key(request.set_version, request.answer_key)
        
        if success:
            return {
                "success": True,
                "message": f"Answer key for {request.set_version} saved successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save answer key")
            
    except Exception as e:
        logger.error(f"Error saving answer key: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/answer-keys")
async def get_answer_keys():
    """
    Get all available answer keys.
    
    Returns:
        Dictionary of all answer keys
    """
    try:
        answer_keys = db_manager.get_all_answer_keys()
        return {
            "success": True,
            "answer_keys": answer_keys
        }
    except Exception as e:
        logger.error(f"Error getting answer keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/answer-keys/{set_version}")
async def get_answer_key(set_version: str):
    """
    Get answer key for a specific set version.
    
    Args:
        set_version: Set version (e.g., 'SET_A')
    
    Returns:
        Answer key for the specified version
    """
    answer_key = db_manager.get_answer_key(set_version)
    
    if not answer_key:
        raise HTTPException(status_code=404, detail="Answer key not found")
    
    return {
        "success": True,
        "set_version": set_version,
        "answer_key": answer_key
    }

@app.get("/api/v1/export/csv")
async def export_results_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """
    Export results to CSV file.
    
    Args:
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)
    
    Returns:
        CSV file download
    """
    try:
        # Convert date strings
        date_from_obj = None
        date_to_obj = None
        
        if date_from:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
        if date_to:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
        
        # Generate export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f"exports/omr_results_{timestamp}.csv"
        
        # Export data
        success = db_manager.export_results_to_csv(
            export_path,
            date_from=date_from_obj,
            date_to=date_to_obj
        )
        
        if success and os.path.exists(export_path):
            return FileResponse(
                path=export_path,
                filename=f"omr_results_{timestamp}.csv",
                media_type='text/csv'
            )
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)