import os
import shutil
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('omr_system')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and limit length
    filename = ' '.join(filename.split())  # Remove extra spaces
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:190] + ext
    
    return filename

def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase.
    
    Args:
        filename: File name
        
    Returns:
        File extension without dot
    """
    return os.path.splitext(filename)[1][1:].lower()

def is_image_file(filename: str) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        filename: File name
        
    Returns:
        True if supported image format
    """
    image_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    return get_file_extension(filename) in image_extensions

def is_pdf_file(filename: str) -> bool:
    """
    Check if file is a PDF.
    
    Args:
        filename: File name
        
    Returns:
        True if PDF file
    """
    return get_file_extension(filename) == 'pdf'

def generate_unique_filename(base_path: str, filename: str) -> str:
    """
    Generate a unique filename if file already exists.
    
    Args:
        base_path: Directory path
        filename: Desired filename
        
    Returns:
        Unique filename
    """
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        return filename
    
    name, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(base_path, new_filename)
        
        if not os.path.exists(new_file_path):
            return new_filename
        
        counter += 1

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        filepath: File path
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None

def convert_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert results to pandas DataFrame for easier manipulation.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Pandas DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    # Flatten the results for DataFrame conversion
    flattened_results = []
    
    for result in results:
        if isinstance(result, dict) and 'results' in result:
            # Extract individual question results
            for question_result in result['results']:
                flattened_result = {
                    'timestamp': result.get('timestamp'),
                    'student_id': result.get('student_info', {}).get('id'),
                    'student_name': result.get('student_info', {}).get('name'),
                    'sheet_version': result.get('sheet_version'),
                    'question_number': question_result.question_number,
                    'selected_answer': question_result.selected_answer,
                    'confidence': question_result.confidence,
                    'is_multiple_selection': question_result.is_multiple_selection,
                    'is_ambiguous': question_result.is_ambiguous
                }
                
                # Add score information if available
                if result.get('score_summary'):
                    score_info = result['score_summary']
                    flattened_result.update({
                        'total_score': score_info.get('raw_score'),
                        'accuracy_percentage': score_info.get('accuracy_percentage')
                    })
                
                flattened_results.append(flattened_result)
        else:
            flattened_results.append(result)
    
    return pd.DataFrame(flattened_results)

def export_to_csv(results: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Export results to CSV file.
    
    Args:
        results: Results data
        filepath: Output file path
        
    Returns:
        Success status
    """
    try:
        df = convert_results_to_dataframe(results)
        
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error exporting to CSV {filepath}: {e}")
        return False

def export_to_excel(results: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Export results to Excel file with multiple sheets.
    
    Args:
        results: Results data
        filepath: Output file path
        
    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        df = convert_results_to_dataframe(results)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary sheet if we have score data
            if 'accuracy_percentage' in df.columns:
                summary_df = df.groupby(['student_id', 'student_name']).agg({
                    'accuracy_percentage': 'first',
                    'total_score': 'first',
                    'question_number': 'count'
                }).rename(columns={'question_number': 'questions_answered'})
                
                summary_df.to_excel(writer, sheet_name='Summary')
        
        return True
    except Exception as e:
        print(f"Error exporting to Excel {filepath}: {e}")
        return False

def calculate_processing_time(start_time: datetime, end_time: datetime = None) -> float:
    """
    Calculate processing time in seconds.
    
    Args:
        start_time: Start time
        end_time: End time (current time if None)
        
    Returns:
        Processing time in seconds
    """
    if end_time is None:
        end_time = datetime.now()
    
    return (end_time - start_time).total_seconds()

def format_processing_time(seconds: float) -> str:
    """
    Format processing time as human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def validate_image_file(filepath: str) -> Dict[str, Any]:
    """
    Validate image file and return information about it.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Validation results
    """
    result = {
        'is_valid': False,
        'file_exists': False,
        'is_image': False,
        'file_size': 0,
        'error_message': None
    }
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            result['error_message'] = "File does not exist"
            return result
        
        result['file_exists'] = True
        
        # Check file size
        file_size = os.path.getsize(filepath)
        result['file_size'] = file_size
        
        if file_size == 0:
            result['error_message'] = "File is empty"
            return result
        
        # Check if it's an image file
        if not is_image_file(filepath):
            result['error_message'] = "File is not a supported image format"
            return result
        
        result['is_image'] = True
        
        # Try to read with OpenCV to validate
        import cv2
        image = cv2.imread(filepath)
        if image is None:
            result['error_message'] = "Cannot read image file (corrupted or unsupported)"
            return result
        
        result['is_valid'] = True
        result['image_shape'] = image.shape
        
    except Exception as e:
        result['error_message'] = f"Error validating file: {str(e)}"
    
    return result

def create_backup(source_path: str, backup_dir: str) -> bool:
    """
    Create backup of a file or directory.
    
    Args:
        source_path: Path to backup
        backup_dir: Backup directory
        
    Returns:
        Success status
    """
    try:
        ensure_directory_exists(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = os.path.basename(source_path)
        backup_name = f"{source_name}_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        if os.path.isfile(source_path):
            shutil.copy2(source_path, backup_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, backup_path)
        else:
            return False
        
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        System information dictionary
    """
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'system': platform.system(),
        'release': platform.release()
    }

def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0