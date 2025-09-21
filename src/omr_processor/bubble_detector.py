import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.cluster import KMeans
import math

class BubbleDetector:
    """
    Detects and extracts bubble locations from OMR sheets using computer vision techniques.
    """
    
    def __init__(self, bubble_area_range: Tuple[int, int] = (50, 500)):
        """
        Initialize bubble detector.
        
        Args:
            bubble_area_range: Min and max area for valid bubbles
        """
        self.min_bubble_area = bubble_area_range[0]
        self.max_bubble_area = bubble_area_range[1]
        self.bubble_aspect_ratio_tolerance = 0.3  # Allow some deviation from perfect circle
        
    def detect_bubbles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all bubbles in the OMR sheet.
        
        Args:
            image: Preprocessed OMR sheet image
            
        Returns:
            List of detected bubbles with their properties
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply threshold
        thresh = self._apply_threshold(gray)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and classify contours as bubbles
        bubbles = []
        for i, contour in enumerate(contours):
            bubble_info = self._analyze_contour(contour, i)
            if bubble_info and self._is_valid_bubble(bubble_info):
                bubbles.append(bubble_info)
        
        # Sort bubbles by position (top to bottom, left to right)
        bubbles = self._sort_bubbles(bubbles)
        
        return bubbles
    
    def _apply_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply adaptive threshold to detect bubbles."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Use adaptive threshold for better results with varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _analyze_contour(self, contour: np.ndarray, contour_id: int) -> Dict[str, Any]:
        """
        Analyze a contour to extract bubble properties.
        
        Args:
            contour: OpenCV contour
            contour_id: Unique identifier for the contour
            
        Returns:
            Dictionary with bubble properties or None if not a valid bubble
        """
        # Calculate basic properties
        area = cv2.contourArea(contour)
        if area < self.min_bubble_area or area > self.max_bubble_area:
            return None
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
            
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Calculate extent (ratio of contour area to bounding rectangle area)
        extent = float(area) / (w * h) if (w * h) != 0 else 0
        
        # Calculate solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        return {
            'id': contour_id,
            'contour': contour,
            'center': (center_x, center_y),
            'area': area,
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'radius': math.sqrt(area / math.pi)  # Approximate radius
        }
    
    def _is_valid_bubble(self, bubble_info: Dict[str, Any]) -> bool:
        """
        Check if a detected shape is a valid bubble based on geometric properties.
        
        Args:
            bubble_info: Bubble information dictionary
            
        Returns:
            True if valid bubble, False otherwise
        """
        # Check aspect ratio (should be close to 1 for circles)
        aspect_ratio = bubble_info['aspect_ratio']
        if abs(aspect_ratio - 1.0) > self.bubble_aspect_ratio_tolerance:
            return False
            
        # Check extent (filled circles should have extent > 0.6)
        if bubble_info['extent'] < 0.6:
            return False
            
        # Check solidity (circles should be fairly solid)
        if bubble_info['solidity'] < 0.8:
            return False
            
        return True
    
    def _sort_bubbles(self, bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort bubbles by position (top to bottom, left to right).
        
        Args:
            bubbles: List of bubble information dictionaries
            
        Returns:
            Sorted list of bubbles
        """
        if not bubbles:
            return bubbles
            
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        return sorted(bubbles, key=lambda b: (b['center'][1], b['center'][0]))
    
    def group_bubbles_into_rows(self, bubbles: List[Dict[str, Any]], 
                               tolerance: int = 20) -> List[List[Dict[str, Any]]]:
        """
        Group bubbles into rows based on their y-coordinates.
        
        Args:
            bubbles: List of detected bubbles
            tolerance: Y-coordinate tolerance for grouping bubbles into same row
            
        Returns:
            List of bubble rows
        """
        if not bubbles:
            return []
            
        # Group bubbles by similar y-coordinates
        rows = []
        current_row = [bubbles[0]]
        
        for i in range(1, len(bubbles)):
            current_bubble = bubbles[i]
            last_bubble_in_row = current_row[-1]
            
            # If y-coordinates are close, add to current row
            if abs(current_bubble['center'][1] - last_bubble_in_row['center'][1]) <= tolerance:
                current_row.append(current_bubble)
            else:
                # Start new row
                # Sort current row by x-coordinate
                current_row.sort(key=lambda b: b['center'][0])
                rows.append(current_row)
                current_row = [current_bubble]
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda b: b['center'][0])
            rows.append(current_row)
        
        return rows
    
    def classify_bubble_fill(self, image: np.ndarray, bubble: Dict[str, Any], 
                           threshold_ratio: float = 0.3) -> Tuple[bool, float]:
        """
        Classify whether a bubble is filled or not.
        
        Args:
            image: Original grayscale image
            bubble: Bubble information dictionary
            threshold_ratio: Minimum ratio of dark pixels to consider bubble as filled
            
        Returns:
            Tuple of (is_filled, fill_ratio)
        """
        # Extract the region of interest
        x, y, w, h = bubble['bounding_box']
        roi = image[y:y+h, x:x+w]
        
        # Create mask for the bubble
        mask = np.zeros(roi.shape, dtype=np.uint8)
        
        # Adjust contour coordinates relative to ROI
        adjusted_contour = bubble['contour'].copy()
        adjusted_contour[:, 0, 0] -= x
        adjusted_contour[:, 0, 1] -= y
        
        # Fill the mask
        cv2.fillPoly(mask, [adjusted_contour], 255)
        
        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        # Calculate the ratio of dark pixels
        total_pixels = np.count_nonzero(mask)
        if total_pixels == 0:
            return False, 0.0
            
        # Count dark pixels (below median intensity in the ROI)
        roi_pixels = masked_roi[mask > 0]
        median_intensity = np.median(roi_pixels)
        dark_pixels = np.count_nonzero(roi_pixels < median_intensity * 0.7)
        
        fill_ratio = dark_pixels / total_pixels
        is_filled = fill_ratio >= threshold_ratio
        
        return is_filled, fill_ratio
    
    def detect_answer_bubbles_grid(self, image: np.ndarray, 
                                  expected_questions: int = 100,
                                  options_per_question: int = 4) -> Dict[str, Any]:
        """
        Detect and organize bubbles in a standard OMR grid format.
        
        Args:
            image: Preprocessed OMR sheet image
            expected_questions: Expected number of questions
            options_per_question: Number of options per question (A, B, C, D)
            
        Returns:
            Dictionary with organized bubble grid information
        """
        # Detect all bubbles
        all_bubbles = self.detect_bubbles(image)
        
        if len(all_bubbles) == 0:
            return {'success': False, 'error': 'No bubbles detected'}
        
        # Group bubbles into rows
        bubble_rows = self.group_bubbles_into_rows(all_bubbles)
        
        # Validate grid structure
        expected_total_bubbles = expected_questions * options_per_question
        actual_total_bubbles = len(all_bubbles)
        
        success = abs(expected_total_bubbles - actual_total_bubbles) <= (expected_questions * 0.1)
        
        return {
            'success': success,
            'total_bubbles': actual_total_bubbles,
            'expected_bubbles': expected_total_bubbles,
            'bubble_rows': bubble_rows,
            'questions_detected': len(bubble_rows),
            'bubbles_per_row': [len(row) for row in bubble_rows],
            'all_bubbles': all_bubbles
        }
    
    def visualize_detected_bubbles(self, image: np.ndarray, 
                                  bubbles: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create visualization of detected bubbles for debugging.
        
        Args:
            image: Original image
            bubbles: List of detected bubbles
            
        Returns:
            Image with bubbles highlighted
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        
        for i, bubble in enumerate(bubbles):
            center = bubble['center']
            radius = int(bubble['radius'])
            
            # Draw circle
            cv2.circle(vis_image, center, radius, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_image, center, 2, (255, 0, 0), -1)
            
            # Add bubble number
            cv2.putText(vis_image, str(i), 
                       (center[0] - 10, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return vis_image