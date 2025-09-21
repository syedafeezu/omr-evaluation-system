import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re
import json

class SheetClassifier:
    """
    Classifies OMR sheets into different versions/sets and identifies key regions.
    """
    
    def __init__(self):
        """Initialize the sheet classifier."""
        self.known_templates = {
            'SET_A': {
                'identifier_region': (50, 50, 200, 100),  # x, y, width, height
                'question_region': (100, 150, 600, 800),
                'total_questions': 100,
                'questions_per_row': 4,
                'identifier_pattern': 'SET.*A'
            },
            'SET_B': {
                'identifier_region': (50, 50, 200, 100),
                'question_region': (100, 150, 600, 800),
                'total_questions': 100,
                'questions_per_row': 4,
                'identifier_pattern': 'SET.*B'
            },
            'SET_C': {
                'identifier_region': (50, 50, 200, 100),
                'question_region': (100, 150, 600, 800),
                'total_questions': 100,
                'questions_per_row': 4,
                'identifier_pattern': 'SET.*C'
            },
            'SET_D': {
                'identifier_region': (50, 50, 200, 100),
                'question_region': (100, 150, 600, 800),
                'total_questions': 100,
                'questions_per_row': 4,
                'identifier_pattern': 'SET.*D'
            }
        }
        
        self.default_template = {
            'identifier_region': (50, 50, 200, 100),
            'question_region': (100, 150, 600, 800),
            'total_questions': 100,
            'questions_per_row': 4
        }
    
    def classify_sheet(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the OMR sheet and identify its version/set.
        
        Args:
            image: Preprocessed OMR sheet image
            
        Returns:
            Dictionary with classification results
        """
        # Try to detect sheet version using OCR/pattern matching
        detected_set = self._detect_sheet_version(image)
        
        # Get template configuration
        template_config = self.known_templates.get(detected_set, self.default_template)
        
        # Validate sheet structure
        validation_results = self._validate_sheet_structure(image, template_config)
        
        # Detect key regions
        regions = self._detect_key_regions(image, template_config)
        
        return {
            'success': True,
            'detected_set': detected_set,
            'confidence': validation_results.get('confidence', 0.8),
            'template_config': template_config,
            'regions': regions,
            'validation': validation_results,
            'recommendations': self._generate_processing_recommendations(validation_results)
        }
    
    def _detect_sheet_version(self, image: np.ndarray) -> Optional[str]:
        """
        Detect sheet version using pattern matching and OCR.
        
        Args:
            image: Input image
            
        Returns:
            Detected sheet version or None
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Define regions to search for identifiers (top portion of the sheet)
        height, width = gray.shape
        search_region = gray[0:int(height*0.2), 0:width]  # Top 20% of image
        
        # Apply OCR preprocessing
        processed_region = self._preprocess_for_ocr(search_region)
        
        # Try to find version indicators using template matching
        detected_version = self._match_version_templates(processed_region)
        
        if detected_version:
            return detected_version
        
        # Fallback: try pattern-based detection
        return self._detect_version_by_patterns(processed_region)
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for better OCR/text detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        # Apply morphological operations to enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Threshold
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Morphological operations
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return processed
    
    def _match_version_templates(self, image: np.ndarray) -> Optional[str]:
        """
        Match against known version templates.
        
        Args:
            image: Preprocessed image region
            
        Returns:
            Detected version or None
        """
        # This is a simplified version - in practice you would have
        # actual template images for each SET version
        
        # For now, we'll use contour-based detection of text regions
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular regions that might contain text
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            # Filter for text-like regions
            if 1.5 < aspect_ratio < 8.0 and area > 100:
                text_regions.append((x, y, w, h))
        
        # If we find text-like regions, assume it's a valid sheet
        # In a real implementation, you would use OCR here
        if text_regions:
            # Default to SET_A for now - in practice, use OCR to read the actual text
            return 'SET_A'
        
        return None
    
    def _detect_version_by_patterns(self, image: np.ndarray) -> Optional[str]:
        """
        Detect version using geometric patterns or other visual cues.
        
        Args:
            image: Input image
            
        Returns:
            Detected version or None
        """
        # Look for specific geometric patterns that might indicate different sets
        # This is a placeholder - implement based on your specific sheet designs
        
        # Count text regions or other distinguishing features
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simple heuristic based on number of detected regions
        valid_regions = [c for c in contours if cv2.contourArea(c) > 50]
        
        if len(valid_regions) > 5:
            return 'SET_A'  # Default fallback
        
        return None
    
    def _validate_sheet_structure(self, image: np.ndarray, 
                                 template_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure of the detected sheet.
        
        Args:
            image: Input image
            template_config: Template configuration
            
        Returns:
            Validation results
        """
        height, width = image.shape[:2]
        
        # Check image dimensions
        dimension_check = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'is_valid_size': width > 500 and height > 600  # Minimum reasonable size
        }
        
        # Check for presence of bubble regions
        bubble_region_check = self._check_bubble_regions(image, template_config)
        
        # Overall confidence calculation
        confidence = 0.0
        if dimension_check['is_valid_size']:
            confidence += 0.3
        if bubble_region_check['bubbles_detected'] > 50:  # Reasonable number of bubbles
            confidence += 0.4
        if 0.6 < dimension_check['aspect_ratio'] < 1.8:  # Reasonable aspect ratio
            confidence += 0.3
        
        return {
            'confidence': min(confidence, 1.0),
            'dimension_check': dimension_check,
            'bubble_region_check': bubble_region_check,
            'is_valid': confidence > 0.6
        }
    
    def _check_bubble_regions(self, image: np.ndarray, 
                             template_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for the presence of bubble regions in expected areas.
        
        Args:
            image: Input image
            template_config: Template configuration
            
        Returns:
            Bubble region check results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Extract question region
        qr = template_config['question_region']
        height, width = gray.shape
        
        # Safely extract region with bounds checking
        x1 = max(0, min(qr[0], width-1))
        y1 = max(0, min(qr[1], height-1))
        x2 = max(x1+1, min(qr[0] + qr[2], width))
        y2 = max(y1+1, min(qr[1] + qr[3], height))
        
        question_region = gray[y1:y2, x1:x2]
        
        # Apply threshold to detect bubbles
        thresh = cv2.adaptiveThreshold(
            question_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find circular-like contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for bubble-like shapes
        bubble_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 500:  # Reasonable bubble size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:  # Roughly circular
                    bubble_count += 1
        
        return {
            'bubbles_detected': bubble_count,
            'expected_bubbles': template_config.get('total_questions', 100) * template_config.get('questions_per_row', 4),
            'region_extracted': True,
            'region_dimensions': (x2-x1, y2-y1)
        }
    
    def _detect_key_regions(self, image: np.ndarray, 
                           template_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect and define key regions of the OMR sheet.
        
        Args:
            image: Input image
            template_config: Template configuration
            
        Returns:
            Dictionary of detected regions
        """
        height, width = image.shape[:2]
        
        # Standard regions based on typical OMR sheet layout
        regions = {
            'header': {
                'coordinates': (0, 0, width, int(height * 0.15)),
                'description': 'Header section with student info and instructions'
            },
            'student_info': {
                'coordinates': (0, int(height * 0.05), width, int(height * 0.1)),
                'description': 'Student name, roll number, etc.'
            },
            'question_grid': {
                'coordinates': template_config['question_region'],
                'description': 'Main question answer grid'
            },
            'footer': {
                'coordinates': (0, int(height * 0.9), width, int(height * 0.1)),
                'description': 'Footer section'
            }
        }
        
        # Validate that regions are within image boundaries
        for region_name, region_info in regions.items():
            coords = region_info['coordinates']
            x, y, w, h = coords
            
            # Clamp coordinates to image boundaries
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = max(1, min(w, width-x))
            h = max(1, min(h, height-y))
            
            regions[region_name]['coordinates'] = (x, y, w, h)
            regions[region_name]['is_valid'] = (x + w <= width and y + h <= height)
        
        return regions
    
    def _generate_processing_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate processing recommendations based on validation results.
        
        Args:
            validation_results: Results from sheet validation
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        confidence = validation_results.get('confidence', 0)
        
        if confidence < 0.5:
            recommendations.append("Sheet quality is poor - consider re-scanning with better lighting")
        elif confidence < 0.7:
            recommendations.append("Sheet quality is marginal - processing may have errors")
        
        dimension_check = validation_results.get('dimension_check', {})
        if not dimension_check.get('is_valid_size', True):
            recommendations.append("Image resolution is too low - use higher quality scan")
        
        bubble_check = validation_results.get('bubble_region_check', {})
        expected_bubbles = bubble_check.get('expected_bubbles', 400)
        detected_bubbles = bubble_check.get('bubbles_detected', 0)
        
        if detected_bubbles < expected_bubbles * 0.5:
            recommendations.append("Low number of bubbles detected - check image quality and contrast")
        
        if not recommendations:
            recommendations.append("Sheet appears to be in good condition for processing")
        
        return recommendations
    
    def get_answer_key_for_set(self, set_version: str) -> Dict[int, str]:
        """
        Get the answer key for a specific set version.
        
        Args:
            set_version: Version of the sheet (e.g., 'SET_A')
            
        Returns:
            Dictionary mapping question numbers to correct answers
        """
        # This would typically load from a configuration file or database
        # For demo purposes, we'll return a sample answer key
        
        answer_keys = {
            'SET_A': self._generate_sample_answer_key('A'),
            'SET_B': self._generate_sample_answer_key('B'), 
            'SET_C': self._generate_sample_answer_key('C'),
            'SET_D': self._generate_sample_answer_key('D')
        }
        
        return answer_keys.get(set_version, answer_keys['SET_A'])
    
    def _generate_sample_answer_key(self, set_type: str) -> Dict[int, str]:
        """
        Generate a sample answer key for demonstration.
        
        Args:
            set_type: Type of the set
            
        Returns:
            Sample answer key
        """
        # Create a pattern based on set type for demo
        patterns = {
            'A': ['A', 'B', 'C', 'D'],
            'B': ['B', 'C', 'D', 'A'],
            'C': ['C', 'D', 'A', 'B'],
            'D': ['D', 'A', 'B', 'C']
        }
        
        pattern = patterns.get(set_type, patterns['A'])
        answer_key = {}
        
        for i in range(1, 101):  # 100 questions
            answer_key[i] = pattern[(i-1) % 4]
        
        return answer_key
    
    def load_answer_keys_from_file(self, filepath: str) -> Dict[str, Dict[int, str]]:
        """
        Load answer keys from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing answer keys
            
        Returns:
            Dictionary of answer keys for different sets
        """
        try:
            with open(filepath, 'r') as f:
                answer_keys = json.load(f)
            
            # Convert string keys to integers for question numbers
            converted_keys = {}
            for set_name, answers in answer_keys.items():
                converted_keys[set_name] = {int(k): v for k, v in answers.items()}
            
            return converted_keys
        except Exception as e:
            print(f"Error loading answer keys from {filepath}: {str(e)}")
            return {}
    
    def save_answer_keys_to_file(self, answer_keys: Dict[str, Dict[int, str]], 
                                filepath: str) -> bool:
        """
        Save answer keys to a JSON file.
        
        Args:
            answer_keys: Dictionary of answer keys
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            # Convert integer keys to strings for JSON serialization
            converted_keys = {}
            for set_name, answers in answer_keys.items():
                converted_keys[set_name] = {str(k): v for k, v in answers.items()}
            
            with open(filepath, 'w') as f:
                json.dump(converted_keys, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving answer keys to {filepath}: {str(e)}")
            return False