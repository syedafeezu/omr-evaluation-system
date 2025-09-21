import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, List
import imutils

class ImagePreprocessor:
    """
    Handles image preprocessing for OMR sheets including rotation correction,
    perspective transformation, and noise reduction.
    """
    
    def __init__(self):
        self.debug_mode = False
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Main preprocessing pipeline for OMR sheet images.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        image = self._resize_image(image)
        image = self._reduce_noise(image)
        image = self._correct_perspective(image)
        image = self._enhance_contrast(image)
        
        return image
    
    def _resize_image(self, image: np.ndarray, max_width: int = 800) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct perspective distortion in OMR sheets.
        
        Args:
            image: Input image
            
        Returns:
            Perspective-corrected image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find the largest rectangular contour (likely the OMR sheet)
        sheet_contour = None
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If contour has 4 points, it's likely rectangular
            if len(approx) == 4:
                sheet_contour = approx
                break
        
        if sheet_contour is not None:
            # Apply perspective transformation
            return self._four_point_transform(image, sheet_contour.reshape(4, 2))
        else:
            # If no rectangular contour found, return original image
            return image
    
    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply four-point perspective transformation.
        
        Args:
            image: Input image
            pts: Four corner points of the region to transform
            
        Returns:
            Transformed image
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width and height of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Set destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # Apply perspective transformation
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left point has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference, bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge((l_channel, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def detect_rotation(self, image: np.ndarray) -> float:
        """
        Detect rotation angle of the image using Hough line transformation.
        
        Args:
            image: Input image
            
        Returns:
            Rotation angle in degrees
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            # Return median angle
            return np.median(angles)
        
        return 0.0
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        if abs(angle) < 0.1:  # Skip rotation for very small angles
            return image
            
        # Get image dimensions
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Perform rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated