"""
Enhanced Spheroid Sizer 
========================
Features:
- Automatic spheroid detection with multiple strategies
- Manual and hybrid measurement modes
- Folder batch processing
- Comprehensive reanalysis table
- Excel export 
"""

import numpy as np
import pandas as pd
import math
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk, colorchooser
import os
import cv2
import re
from PIL import Image, ImageTk, ImageDraw, ImageOps
from datetime import datetime
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import pyperclip
import webbrowser
from tkinter.scrolledtext import ScrolledText

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    area: float = 0.0
    roundness: float = 0.0
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    perimeter: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    mode: str = ""
    success: bool = True
    error_message: str = ""
    confidence_score: float = 0.0

class EnhancedImageProcessor:
    def __init__(self):
        # More flexible parameters for better spheroid detection
        self.min_area = 1000  # Reduced to catch smaller spheroids
        self.max_area = 5000000  # Increased for larger spheroids
        self.min_roundness = 0.15  # More permissive for irregular spheroids
        self.min_solidity = 0.4   # More permissive for complex shapes
        self.min_extent = 0.3     # More flexible for various shapes
        self.max_aspect_ratio = 3.0  # Allow more elongated shapes

        # Stitching parameters
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        
    def enhanced_detection(self, image_path: str, preview_callback=None):
        """Enhanced automatic detection with multiple strategies"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Try multiple detection strategies
            strategies = [
                self._strategy_adaptive_threshold,
                self._strategy_otsu_with_morphology,
                self._strategy_edge_based,
                self._strategy_watershed,
                self._strategy_contour_approximation
            ]
            
            best_contour = None
            best_result = None
            best_confidence = 0.0
            
            for i, strategy in enumerate(strategies):
                try:
                    if preview_callback:
                        preview_callback(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
                                       f"Trying strategy {i+1}/{len(strategies)}")
                    
                    contours = strategy(gray)
                    
                    if contours:
                        # Evaluate each contour
                        for contour in contours:
                            if self._is_valid_spheroid_flexible(contour, gray.shape):
                                result = self._calculate_comprehensive_metrics(contour)
                                
                                if result.confidence_score > best_confidence:
                                    best_confidence = result.confidence_score
                                    best_contour = contour
                                    best_result = result
                                    
                                    if preview_callback:
                                        preview_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                                        cv2.drawContours(preview_img, [contour], -1, (0, 255, 0), 2)
                                        preview_callback(preview_img, 
                                                       f"Found candidate (confidence: {result.confidence_score:.2f})")
                
                except Exception as e:
                    logger.warning(f"Strategy {i+1} failed: {e}")
                    continue
            
            if best_contour is not None and best_result is not None:
                best_result.mode = 'automatic'
                best_result.success = True
                return best_contour, best_result
            
            return None, "No valid spheroids detected with any strategy"
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None, str(e)
    
    def save_processed_image(self, original_path: str, contour, result: AnalysisResult, output_folder: str):
        """Save processed image with contour overlay"""
        try:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Load original image
            img = cv2.imread(original_path)
            if img is None:
                return None
            
            # Create filename with mode suffix
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            if result.mode == 'manual' or result.mode == 'hybrid':
                output_filename = f"{base_name}-manual.tif"
            else:
                output_filename = f"{base_name}-detected.tif"
            
            output_path = os.path.join(output_folder, output_filename)
            
            # Draw contour on image
            processed_img = img.copy()
            
            # Choose color based on mode
            if result.mode == 'manual' or result.mode == 'hybrid':
                color = (255, 0, 0)  # Blue for manual
            else:
                color = (0, 255, 0)  # Green for automatic
            
            # Draw contour with thickness proportional to image size
            thickness = max(2, min(img.shape[:2]) // 200)
            cv2.drawContours(processed_img, [contour], -1, color, thickness)
            
            # Add confidence text for automatic detection
            if result.mode == 'automatic' and hasattr(result, 'confidence_score'):
                font_scale = min(img.shape[:2]) / 1000
                text = f"Conf: {result.confidence_score:.2f}"
                cv2.putText(processed_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), 2)
                cv2.putText(processed_img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 0, 0), 1)
            
            # Save processed image
            cv2.imwrite(output_path, processed_img)
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            return None
    
    def _strategy_adaptive_threshold(self, gray):
        """Strategy 1: Adaptive thresholding"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple adaptive threshold methods
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 5)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 5)
        
        # Combine thresholds
        combined = cv2.bitwise_or(thresh1, thresh2)
        inverted = cv2.bitwise_not(combined)
        
        # Light morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _strategy_otsu_with_morphology(self, gray):
        """Strategy 2: Otsu thresholding with morphology"""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresh)
        
        # More aggressive morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _strategy_edge_based(self, gray):
        """Strategy 3: Edge-based detection"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection with multiple parameter sets
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Fill holes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create filled contours
        filled = np.zeros_like(edges)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.fillPoly(filled, [contour], 255)
        
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _strategy_watershed(self, gray):
        """Strategy 4: Watershed segmentation"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create markers
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # Extract contours from markers
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _strategy_contour_approximation(self, gray):
        """Strategy 5: Multiple threshold levels with contour approximation"""
        contours_all = []
        
        # Try multiple threshold levels
        for thresh_val in [50, 80, 120, 150, 180]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Light morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Approximate contours to smooth them
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 5:  # Need at least 5 points for ellipse fitting
                    contours_all.append(approx)
        
        return contours_all
    
    def _is_valid_spheroid_flexible(self, contour, image_shape):
        """More flexible validation for spheroid-like shapes"""
        area = cv2.contourArea(contour)
        
        # Basic area check
        if area < self.min_area or area > self.max_area:
            return False
        
        # Skip very small contours
        if len(contour) < 5:
            return False
        
        # Perimeter check
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return False
        
        # Basic roundness (more permissive)
        roundness = 4 * np.pi * area / (perimeter ** 2)
        if roundness < self.min_roundness:
            return False
        
        # Convex hull checks
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.min_solidity:
                    return False
        except:
            pass  # Skip this check if it fails
        
        # Bounding rectangle checks
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio check
            if min(w, h) > 0:
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > self.max_aspect_ratio:
                    return False
            
            # Extent check
            rect_area = w * h
            if rect_area > 0:
                extent = area / rect_area
                if extent < self.min_extent:
                    return False
        except:
            pass  # Skip this check if it fails
        
        # Position check - prefer central objects but don't exclude edge objects
        height, width = image_shape
        try:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                # Very lenient position check - just exclude objects too close to edges
                edge_margin = 0.05  # 5% margin
                if (cx < width * edge_margin or cx > width * (1 - edge_margin) or
                    cy < height * edge_margin or cy > height * (1 - edge_margin)):
                    # Still allow but with lower confidence
                    pass
        except:
            pass
        
        return True
    
    def _calculate_comprehensive_metrics(self, contour):
        """Calculate comprehensive metrics with error handling"""
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Solidity
            try:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
            except:
                solidity = 0.5
            
            # Axis lengths
            try:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    major_axis = max(w, h)
                    minor_axis = min(w, h)
                    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            except:
                x, y, w, h = cv2.boundingRect(contour)
                major_axis = max(w, h)
                minor_axis = min(w, h)
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            
            # Enhanced confidence score
            confidence = self._calculate_confidence_score(area, roundness, solidity, aspect_ratio)
            
            return AnalysisResult(
                area=area,
                roundness=roundness,
                major_axis_length=major_axis,
                minor_axis_length=minor_axis,
                perimeter=perimeter,
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return AnalysisResult(success=False, error_message=str(e))
    
    def _calculate_confidence_score(self, area, roundness, solidity, aspect_ratio):
        """Calculate confidence score based on multiple factors"""
        try:
            # Size confidence (prefer medium-large spheroids)
            if area < 1000:
                size_conf = 0.3
            elif area < 10000:
                size_conf = 0.7
            elif area < 500000:
                size_conf = 1.0
            elif area < 20000000:
                size_conf = 0.9
            else:
                size_conf = 0.7
            
            # Shape confidence
            roundness_conf = min(roundness * 2.5, 1.0)  # Scale roundness
            solidity_conf = min(solidity * 1.2, 1.0)    # Scale solidity
            
            # Aspect ratio confidence (prefer closer to 1, but allow some elongation)
            if aspect_ratio <= 2.0:
                aspect_conf = 1.0
            elif aspect_ratio <= 3.0:
                aspect_conf = 0.8
            elif aspect_ratio <= 4.0:
                aspect_conf = 0.6
            else:
                aspect_conf = 0.4
            
            # Weighted combination
            confidence = (size_conf * 0.3 + 
                         roundness_conf * 0.3 + 
                         solidity_conf * 0.25 + 
                         aspect_conf * 0.15)
            
            return min(confidence, 1.0)
            
        except:
            return 0.5
    
    def hybrid_detection_around_points(self, image_path: str, manual_points: List[Tuple[int, int]], 
                                     search_radius: int = 50):
        """Hybrid detection in areas around manual points with multiple methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Create region of interest around manual points
            mask = np.zeros(gray.shape, dtype=np.uint8)
            for point in manual_points:
                cv2.circle(mask, point, search_radius, 255, -1)
            
            # Try multiple detection methods in the ROI
            best_contour = None
            best_distance = float('inf')
            
            # Method 1: Threshold-based
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            for thresh_method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C]:
                try:
                    if thresh_method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
                        thresh = cv2.adaptiveThreshold(blurred, 255, thresh_method, cv2.THRESH_BINARY, 15, 5)
                    else:
                        _, thresh = cv2.threshold(blurred, 0, 255, thresh_method)
                    
                    thresh_inv = cv2.bitwise_not(thresh)
                    masked = cv2.bitwise_and(thresh_inv, mask)
                    
                    # Light morphology
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    cleaned = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel, iterations=1)
                    
                    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < 50:
                            continue
                        
                        # Calculate minimum distance to manual points
                        min_dist = float('inf')
                        for point in manual_points:
                            dist = abs(cv2.pointPolygonTest(contour, point, True))
                            min_dist = min(min_dist, dist)
                        
                        if min_dist < best_distance:
                            best_distance = min_dist
                            best_contour = contour
                
                except Exception as e:
                    logger.warning(f"Hybrid detection method failed: {e}")
                    continue
            
            # Method 2: Edge-based in ROI
            try:
                edges = cv2.Canny(blurred, 30, 100)
                edges_masked = cv2.bitwise_and(edges, mask)
                
                # Dilate and close
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges_closed = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue
                    
                    min_dist = float('inf')
                    for point in manual_points:
                        dist = abs(cv2.pointPolygonTest(contour, point, True))
                        min_dist = min(min_dist, dist)
                    
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_contour = contour
            
            except Exception as e:
                logger.warning(f"Edge-based hybrid detection failed: {e}")
            
            if best_contour is not None:
                result = self._calculate_comprehensive_metrics(best_contour)
                result.mode = 'hybrid'
                result.success = True
                return best_contour, result
            
            return None, "No suitable contour found near manual points"
            
        except Exception as e:
            logger.error(f"Hybrid detection error: {e}")
            return None, str(e)

    def align_images(self, img1, img2, method='feature'):
        """Align two images using specified method"""
        if method == 'feature':
            # Find keypoints and descriptors
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                matches = self.flann.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]
                
                if len(good) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    aligned = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
                    return aligned, M
        return img1, None

    def stitch_images(self, img1, img2, method='auto'):
        """Stitch two images with specified method"""
        # Ensure images are valid
        if img1 is None or img2 is None:
            raise ValueError("Invalid input images")
    
        methods = {
            'horizontal': lambda a, b: (np.hstack((a, b))), 
            'vertical': lambda a, b: (np.vstack((a, b))),
            'horizontal_mirror': lambda a, b: (np.hstack((a, cv2.flip(b, 1)))),
            'vertical_mirror': lambda a, b: (np.vstack((a, cv2.flip(b, 0)))),
            'feature': self._stitch_with_features
        }
    
        if method == 'auto':
            best_result = None
            best_score = -1
        
            for m in ['feature', 'horizontal', 'vertical', 'horizontal_mirror', 'vertical_mirror']:
                try:
                    stitched = methods[m](img1, img2)
                    if stitched is not None:
                        if isinstance(stitched, tuple):  # Handle feature method return
                            stitched = stitched[0]
                        score = self._evaluate_stitch_quality(stitched)
                        if score > best_score:
                            best_score = score
                            best_result = stitched
                except Exception as e:
                    logger.warning(f"Stitching method {m} failed: {e}")
                    continue
        
            return best_result if best_result is not None else img1
    
        try:
            result = methods[method](img1, img2)
            return result[0] if isinstance(result, tuple) else result
        except Exception as e:
            logger.error(f"Stitching failed with method {method}: {e}")
            return img1

    def _stitch_with_features(self, img1, img2):
        """Stitch images using feature matching"""
        try:
            # Resize images if they're too large
            max_dim = 2000
            h, w = img1.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img1 = cv2.resize(img1, (int(w*scale), int(h*scale)))
            h, w = img2.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img2 = cv2.resize(img2, (int(w*scale), int(h*scale)))
        
            status, stitched = self.stitcher.stitch([img1, img2])
            if status == cv2.Stitcher_OK:
                return stitched
            raise ValueError(f"Feature-based stitching failed with status {status}")
        except Exception as e:
            raise ValueError(f"Feature-based stitching error: {str(e)}")
    
    def _evaluate_stitch_quality(self, stitched_img):
        """Evaluate quality of stitched image"""
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Count non-black pixels
        non_zero = cv2.countNonZero(thresh)
        coverage = non_zero / (gray.shape[0] * gray.shape[1])
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / non_zero if non_zero > 0 else 0
        
        return coverage * 0.7 + edge_density * 0.3

def calculate_volume(length_um, width_um):
    """Calculate volume based on the shape (sphere or prolate spheroid)"""
    if width_um == 0:
        return 0.0
    shape_ratio = abs(length_um - width_um) / width_um
    if shape_ratio <= 0.05:
        radius = length_um / 2
        volume = (4/3) * math.pi * radius**3
    else:
        a = length_um / 2
        b = width_um / 2
        volume = (4/3) * math.pi * a * b**2
    return volume / 1e9  # µm³ to mm³

class StitchingPreviewWindow:
    """Interactive window for previewing stitching results"""
    def __init__(self, parent, processor):
        self.parent = parent
        self.processor = processor
        self.window = tk.Toplevel(parent)
        self.window.title("Stitching Preview")
        self.window.geometry("1200x800")
        
        # Create frames for layout
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display frame
        image_frame = tk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for each image
        self.canvas1 = tk.Canvas(image_frame, width=400, height=400)
        self.canvas1.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas2 = tk.Canvas(image_frame, width=400, height=400)
        self.canvas2.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.result_canvas = tk.Canvas(image_frame, width=400, height=400)
        self.result_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.method_var = tk.StringVar(value='auto')
        self.blend_var = tk.BooleanVar(value=True)
        
        # Method selection
        method_frame = tk.Frame(control_frame)
        method_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        methods = ['auto', 'horizontal', 'vertical', 'horizontal_mirror', 'vertical_mirror', 'feature']
        tk.OptionMenu(method_frame, self.method_var, *methods).pack(side=tk.LEFT)
        
        # Options
        options_frame = tk.Frame(control_frame)
        options_frame.pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(options_frame, text="Blend Seam", variable=self.blend_var).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Update Preview", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Apply", command=self.apply_stitching).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Image references
        self.img1 = None
        self.img2 = None
        self.result_image = None
        self.photo1 = None
        self.photo2 = None
        self.photo_result = None

    def show_images(self, img1, img2):
        """Load images for preview"""
        try:
            self.img1 = img1
            self.img2 = img2
            
            # Display input images
            self._display_image(self.canvas1, img1, "Image 1")
            self._display_image(self.canvas2, img2, "Image 2")
            
            # Generate initial preview
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")

    def update_preview(self):
        """Update the preview with current settings"""
        try:
            if self.img1 is None or self.img2 is None:
                raise ValueError("No images loaded for stitching")
            
            method = self.method_var.get()
            self._display_loading(self.result_canvas)
            self.window.update_idletasks()  # Force UI update
        
            # Perform stitching
            stitched = self.processor.stitch_images(self.img1, self.img2, method)
        
            if stitched is None:
                raise ValueError("Stitching returned no result")
            
            if self.blend_var.get():
                stitched = self._blend_seam(stitched)
            
            self.result_image = stitched
            self._display_image(self.result_canvas, stitched, "Stitched Result")
        
        except Exception as e:
            messagebox.showerror("Preview Error", str(e))
            self._display_error(self.result_canvas)

    def _display_image(self, canvas, image, title=""):
        """Display image on canvas"""
        try:
            if image is None:
                raise ValueError("No image provided")
                
            # Convert to RGB if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            img_pil = Image.fromarray(img_rgb)
        
            # Get canvas dimensions (with fallback)
            try:
                canvas_width = max(canvas.winfo_width() - 20, 100)
                canvas_height = max(canvas.winfo_height() - 40, 100)
            except:
                canvas_width, canvas_height = 400, 400
            
            # Calculate aspect ratio preserving dimensions
            img_ratio = img_pil.width / img_pil.height
            canvas_ratio = canvas_width / canvas_height
        
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            
            # Ensure minimum dimensions
            new_width = max(new_width, 10)
            new_height = max(new_height, 10)
        
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
            # Update canvas
            canvas.delete("all")
            photo = ImageTk.PhotoImage(img_pil)
        
            # Keep reference
            if canvas == self.canvas1:
                self.photo1 = photo
            elif canvas == self.canvas2:
                self.photo2 = photo
            else:
                self.photo_result = photo
            
            # Center the image
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2
            canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)
        
            # Add title
            canvas.create_text(canvas_width//2, canvas_height-10, 
                             text=title, fill="black", font=('Arial', 10))
        
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            self._display_error(canvas)

    def _display_loading(self, canvas):
        """Show loading state on canvas"""
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                          text="Processing...", fill="black")

    def _display_error(self, canvas):
        """Show error state on canvas"""
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                          text="Stitching Failed", fill="red")

    def _blend_seam(self, image):
        """Apply seam blending to stitched image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(thresh, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            result = image.copy()
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (mask/255.0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error blending seam: {e}")
            return image

    def apply_stitching(self):
        """Apply current stitching settings"""
        if self.result_image is not None:
            self.window.destroy()

class LivePreviewWindow:
    """Live preview window for detection process"""
    def __init__(self, title="Detection Preview"):
        self.window_name = title
        self.active = False
    
    def start(self):
        self.active = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def update(self, image, status_text="Processing..."):
        if not self.active:
            return
        
        try:
            # Add status text to image
            display_img = image.copy()
            cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, status_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Shadow
            if display_img.shape[0] > 400 or display_img.shape[1] > 400:
                display_img = cv2.resize(display_img, (400, 400))
            cv2.imshow(self.window_name, display_img)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Preview update failed: {e}")
    
    def close(self):
        self.active = False
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass

class EnhancedManualTool:
    """Enhanced manual drawing with hybrid detection"""
    def __init__(self):
        self.points = []
        self.image = None
        self.display_image = None
        self.processor = EnhancedImageProcessor()
        self.window_name = "Manual/Hybrid - Left:Add Point, Right:Remove, H:Hybrid, Enter:Finish, Q:Cancel"
        self.finished = False
        self.cancelled = False
        self.hybrid_mode = False
    
    def manual_analysis(self, image_path: str):
        """Enhanced manual analysis with hybrid option"""
        try:
            self.points = []
            self.finished = False
            self.cancelled = False
            self.hybrid_mode = False
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            if len(img.shape) == 3:
                self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                self.image = img.copy()
            
            # Convert to BGR for display
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.display_image = self.image.copy()
            
            # Setup window
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            
            # Main loop
            while not self.finished and not self.cancelled:
                cv2.imshow(self.window_name, self.display_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    self.cancelled = True
                    break
                elif key == ord('h') or key == ord('H'):
                    if len(self.points) >= 2:
                        self._try_hybrid_detection(image_path)
                elif key == 13 and len(self.points) >= 3:  # Enter
                    self._finish_drawing()
                    break
                elif key == 27:  # Escape
                    self.cancelled = True
                    break
            
            cv2.destroyAllWindows()
            
            if self.cancelled or len(self.points) < 3:
                return None, "Analysis cancelled"
            
            # Calculate metrics
            contour = np.array(self.points, dtype=np.int32)
            result = self.processor._calculate_comprehensive_metrics(contour.reshape(-1, 1, 2))
            result.mode = 'hybrid' if self.hybrid_mode else 'manual'
            result.success = True
            
            return contour, result
            
        except Exception as e:
            cv2.destroyAllWindows()
            logger.error(f"Manual analysis error: {e}")
            return None, str(e)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse callback"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._update_display()
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            self._update_display()
        elif event == cv2.EVENT_MOUSEMOVE and self.points:
            # Show preview line
            temp_display = self.display_image.copy()
            cv2.line(temp_display, self.points[-1], (x, y), (128, 128, 128), 1)
            cv2.imshow(self.window_name, temp_display)
    
    def _update_display(self):
        """Update display with current points"""
        self.display_image = self.image.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 0) if self.hybrid_mode else (0, 0, 255)
            cv2.circle(self.display_image, point, 4, color, -1)
            cv2.putText(self.display_image, str(i+1), (point[0]+8, point[1]-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw lines
        for i in range(len(self.points) - 1):
            color = (0, 255, 0) if self.hybrid_mode else (0, 0, 255)
            cv2.line(self.display_image, self.points[i], self.points[i + 1], color, 2)
        
        # Preview closing line
        if len(self.points) >= 3:
            cv2.line(self.display_image, self.points[-1], self.points[0], (128, 128, 128), 1)
    
    def _try_hybrid_detection(self, image_path: str):
        """Try hybrid detection around current points"""
        try:
            contour, result = self.processor.hybrid_detection_around_points(image_path, self.points, search_radius=80)
            
            if contour is not None and isinstance(result, AnalysisResult) and result.success:
                # Update points with detected contour
                self.points = [tuple(pt[0]) for pt in contour]
                self.hybrid_mode = True
                self._update_display()
                
                # Show result briefly
                temp_display = self.display_image.copy()
                cv2.putText(temp_display, f"Hybrid detection: Area={result.area:.1f}, Conf={result.confidence_score:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_display)
                cv2.waitKey(1500)
            else:
                # Show failure message
                temp_display = self.display_image.copy()
                cv2.putText(temp_display, "Hybrid detection failed - continue manually", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(self.window_name, temp_display)
                cv2.waitKey(1500)
        
        except Exception as e:
            logger.error(f"Hybrid detection error: {e}")
    
    def _finish_drawing(self):
        """Finish the drawing"""
        if len(self.points) >= 3:
            color = (0, 255, 0) if self.hybrid_mode else (0, 0, 255)
            cv2.line(self.display_image, self.points[-1], self.points[0], color, 2)
            cv2.imshow(self.window_name, self.display_image)
            cv2.waitKey(1000)
            self.finished = True

class ReanalysisTable:
    """Reanalysis table with thumbnails, selection, and stitching functionality"""
    
    def __init__(self, parent, processor, manual_tool):
        self.parent = parent
        self.processor = processor
        self.manual_tool = manual_tool
        self.results_data = []
        self.selection_vars = []
        self.photo_refs = []
        self.draw_color = (0, 0, 255)  # Default red for manual drawing

    def _stitch_selected_images(self, results_df, scale, output_folder, parent_window):
        """Stitch selected images with interactive preview"""
        selected_indices = [i for i, data in enumerate(self.results_data) if data['selection_var'].get()]
        if len(selected_indices) != 2:
            messagebox.showwarning("Invalid Selection", "Please select exactly 2 images to stitch")
            return
        
        # Get selected images
        row1 = results_df.iloc[selected_indices[0]]
        row2 = results_df.iloc[selected_indices[1]]
        
        img1_path = row1['image_path']
        img2_path = row2['image_path']
        
        try:
            # Load images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                raise ValueError("Could not load one or both images")
            
            # Show interactive stitching preview
            preview = StitchingPreviewWindow(self.parent, self.processor)
            preview.show_images(img1, img2)
            self.parent.wait_window(preview.window)
            
            if not hasattr(preview, 'result_image') or preview.result_image is None:
                return
            
            stitched_img = preview.result_image
            
            # Save stitched image
            stitch_folder = os.path.join(output_folder, "stitched_images")
            os.makedirs(stitch_folder, exist_ok=True)
            
            base1 = os.path.splitext(os.path.basename(img1_path))[0]
            base2 = os.path.splitext(os.path.basename(img2_path))[0]
            stitch_path = os.path.join(stitch_folder, f"stitched_{base1}_{base2}.tif")
            cv2.imwrite(stitch_path, stitched_img)
            
            # Run detection on stitched image
            contour, result = self.processor.enhanced_detection(stitch_path)
            if contour is None:
                contour, result = self.manual_tool.manual_analysis(stitch_path)
                if contour is None:
                    raise ValueError("Detection failed on stitched image")
            
            # Save processed image
            processed_path = self.processor.save_processed_image(
                stitch_path, contour, result, output_folder)
            
            # Calculate measurements
            area_um2 = result.area * (scale ** 2)
            length_um = result.major_axis_length * scale
            width_um = result.minor_axis_length * scale
            volume_mm3 = calculate_volume(length_um, width_um)
            
            # Create new entry
            new_entry = {
                'Filename': f"stitched_{base1}_{base2}.tif",
                'Folder': stitch_folder,
                'Mode': result.mode,
                'Area_um2': area_um2,
                'Length_um': length_um,
                'Width_um': width_um,
                'Volume_mm3': volume_mm3,
                'Roundness': result.roundness,
                'Solidity': result.solidity,
                'Aspect_Ratio': result.aspect_ratio,
                'Confidence': result.confidence_score,
                'image_path': stitch_path,
                'processed_path': processed_path,
                'stitched_from': f"{row1['Filename']}, {row2['Filename']}",
                'stitch_method': preview.method_var.get(),
                'stitch_blended': preview.blend_var.get()
            }
            
            # Update dataframe
            results_df = results_df.drop(selected_indices).reset_index(drop=True)
            results_df = pd.concat([results_df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Refresh table
            parent_window.destroy()
            self.show_reanalysis_table(results_df, scale, output_folder)
            
            messagebox.showinfo("Success", "Images stitched and analyzed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stitch images:\n{str(e)}")
            logger.error(f"Stitching error: {e}")

    def show_zoom(self, event, img_path):
        zoom_window = tk.Toplevel(self.parent)
        zoom_window.title(f"Zoom View - {os.path.basename(img_path)}")
        zoom_window.geometry("600x600")
        zoom_window.resizable(True, True)

        img = cv2.imread(img_path)
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(img_pil)

        canvas = tk.Canvas(zoom_window, width=photo.width(), height=photo.height())
        canvas.pack()
        canvas.image = photo
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        lens_radius = 40
        lens = canvas.create_oval(0, 0, 0, 0, outline='blue', width=2)

        def move_lens(event):
            x, y = event.x, event.y
            canvas.coords(lens, x - lens_radius, y - lens_radius, x + lens_radius, y + lens_radius)

        canvas.bind('<Motion>', move_lens)
        zoom_window.bind('<Escape>', lambda e: zoom_window.destroy())
        tk.Button(zoom_window, text="Close", command=zoom_window.destroy).pack(pady=5)

          
    def show_reanalysis_table(self, results_df: pd.DataFrame, scale: float, output_folder: str):
        """Show reanalysis table with thumbnails and selection boxes"""
        try:
            # Create reanalysis window
            reanalysis_window = tk.Toplevel(self.parent)
            reanalysis_window.title("Reanalysis Table - Select Images to Reanalyze")
            reanalysis_window.geometry("1200x800")
            reanalysis_window.transient(self.parent)
            
            # Control frame
            control_frame = tk.Frame(reanalysis_window)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Buttons
            btn_frame = tk.Frame(control_frame)
            btn_frame.pack(side=tk.LEFT, padx=5)
            
            tk.Button(btn_frame, text="Select All", 
                     command=lambda: self._select_all(True)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text="Deselect All", 
                     command=lambda: self._select_all(False)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text="Reanalyze Selected", 
                     command=lambda: self._reanalyze_selected(results_df, scale, output_folder, reanalysis_window)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text="Stitch Selected", 
                 command=lambda: self._stitch_selected_images(results_df, scale, output_folder, reanalysis_window)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text="Export to Excel", 
                     command=lambda: self._export_to_excel(results_df)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text="Copy to Clipboard", 
                     command=lambda: self._copy_to_clipboard(results_df)).pack(side=tk.LEFT, padx=2)
            
            # Color selection
            def change_color():
                color_rgb = colorchooser.askcolor(title="Choose drawing color")
                if color_rgb[0]:
                    self.draw_color = tuple(int(c) for c in color_rgb[0][::-1])  # Convert to BGR
                    
            tk.Button(control_frame, text="Change Drawing Color", 
                     command=change_color).pack(side=tk.RIGHT, padx=5)
            
            # Scrollable results frame
            canvas_frame = tk.Frame(reanalysis_window)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Clear previous data
            self.results_data = []
            self.selection_vars = []
            self.photo_refs = []
            
            # Populate table
            for idx, row in results_df.iterrows():
                self._create_table_row(scrollable_frame, row, idx, output_folder)
            
            # Make window modal
            reanalysis_window.grab_set()
            
        except Exception as e:
            logger.error(f"Error showing reanalysis table: {e}")
            messagebox.showerror("Error", f"Failed to show reanalysis table: {str(e)}")
    
    def _create_table_row(self, parent, row, idx, output_folder):
        """Create a row in the reanalysis table matching spheroid_sizer-AG.py style"""
        try:
            # Main row frame
            row_frame = tk.Frame(parent, borderwidth=1, relief="solid", padx=5, pady=5)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Checkbox for reanalysis
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(row_frame, text="Reanalyze", variable=var)
            cb.pack(side=tk.RIGHT, padx=5)
            self.selection_vars.append(var)  # Changed from selected_for_reanalysis to selection_vars
            
            # Get image paths
            filename = os.path.splitext(row['Filename'])[0] if 'Filename' in row else os.path.splitext(row['File'])[0]
            if row['Mode'] == 'manual' or row['Mode'] == 'hybrid':
                processed_img = os.path.join(output_folder, f"{filename}-manual.tif")
            else:
                processed_img = os.path.join(output_folder, f"{filename}-detected.tif")
            
            original_img = row['image_path'] if 'image_path' in row else os.path.join(row['Folder'] if 'Folder' in row else "", row['File'])
            display_img = processed_img if os.path.exists(processed_img) else original_img
            
            # Create thumbnail
            thumbnail = self._create_thumbnail(display_img)
            self.photo_refs.append(thumbnail)
            
            # Image label
            img_label = tk.Label(row_frame, image=thumbnail)
            img_label.pack(side=tk.LEFT)
            img_label.bind("<Button-3>", lambda e, path=original_img: self._show_zoom(e, path))
            
            # Data frame
            data_frame = tk.Frame(row_frame)
            data_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
            
            # File info
            file_frame = tk.Frame(data_frame)
            file_frame.pack(fill=tk.X)
            tk.Label(file_frame, text=f"File: {row['Filename'] if 'Filename' in row else row['File']}", 
                    anchor='w', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
            color = 'blue' if 'manual' in str(row['Mode']).lower() else 'green'
            tk.Label(file_frame, text=f"Mode: {row['Mode']}", 
                    anchor='w', fg=color).pack(side=tk.LEFT, padx=10)
            
            if 'Confidence' in row:
                conf_color = '#27ae60' if row['Confidence'] > 0.7 else '#f39c12' if row['Confidence'] > 0.4 else '#e74c3c'
                tk.Label(file_frame, text=f"Confidence: {row['Confidence']:.3f}", 
                        anchor='w', fg=conf_color).pack(side=tk.LEFT, padx=10)
            # Add stitching info if available
            if 'stitched_from' in row:
                tk.Label(file_frame, text=f"Stitched from: {row['stitched_from']}", 
                        anchor='w', fg='purple').pack(side=tk.LEFT, padx=10)
            
            # Metrics
            metrics_frame = tk.Frame(data_frame)
            metrics_frame.pack(fill=tk.X, pady=2)
            
            metric_font = ('Arial', 9)
            tk.Label(metrics_frame, text=f"Area: {row['Area_um2'] if 'Area_um2' in row else row['Area (µm²)']:.2f} µm²", 
                    width=20, anchor='w', font=metric_font).pack(side=tk.LEFT)
            tk.Label(metrics_frame, text=f"Length: {row['Length_um'] if 'Length_um' in row else row['Length (µm)']:.2f} µm", 
                    width=20, anchor='w', font=metric_font).pack(side=tk.LEFT, padx=5)
            tk.Label(metrics_frame, text=f"Width: {row['Width_um'] if 'Width_um' in row else row['Width (µm)']:.2f} µm", 
                    width=20, anchor='w', font=metric_font).pack(side=tk.LEFT, padx=5)
            tk.Label(metrics_frame, text=f"Volume: {row['Volume_mm3'] if 'Volume_mm3' in row else row['Volume (mm³)']:.6f} mm³", 
                    width=25, anchor='w', font=metric_font).pack(side=tk.LEFT, padx=5)
            
            # Store row data
            self.results_data.append({
                'row_data': row,
                'index': idx,
                'selection_var': var
            })
            
        except Exception as e:
            logger.error(f"Error creating table row: {e}")
    
    def _create_thumbnail(self, image_path: str, size: Tuple[int, int] = (150, 150)):
        """Create thumbnail image for display"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil = ImageOps.expand(img_pil, border=2, fill='gray')
            img_pil.thumbnail(size, Image.Resampling.LANCZOS)
            
            return ImageTk.PhotoImage(img_pil)
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            # Return placeholder
            blank_img = Image.new('RGB', size, (200, 200, 200))
            draw = ImageDraw.Draw(blank_img)
            draw.text((10, 10), "No Image", fill="black")
            return ImageTk.PhotoImage(blank_img)
    
    def _show_zoom(self, event, img_path: str):
        """Show zoomed image in popup window"""
        try:
            zoom_window = tk.Toplevel(self.parent)
            zoom_window.title(f"Zoom View - {os.path.basename(img_path)}")
            zoom_window.geometry("500x500")
            zoom_window.resizable(True, True)
            
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_pil.thumbnail((480, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_pil)
                
                label = tk.Label(zoom_window, image=photo)
                label.pack(padx=10, pady=10)
                
                # Keep reference
                label.image = photo
                
                # Bind close events
                label.bind("<Button-3>", lambda e: zoom_window.destroy())  # Right-click
                zoom_window.bind("<Escape>", lambda e: zoom_window.destroy())  # ESC key
                
                # Focus the window
                zoom_window.focus_set()
            
        except Exception as e:
            logger.error(f"Error showing zoom: {e}")
    
    def _select_all(self, select: bool):
        """Select or deselect all items"""
        for var in self.selection_vars:
            var.set(select)
    
    def _reanalyze_selected(self, results_df, scale, output_folder, parent_window):
        """Reanalyze selected images"""
        selected_indices = [i for i, data in enumerate(self.results_data) if data['selection_var'].get()]
        if not selected_indices:
            messagebox.showwarning("No Selection", "No images selected for reanalysis")
            return
        
        # Create progress window
        progress_window = tk.Toplevel(parent_window)
        progress_window.title("Reanalyzing Images")
        progress_window.geometry("400x120")
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                     maximum=len(selected_indices))
        progress_bar.pack(padx=20, pady=20, fill=tk.X)
        
        progress_label = tk.Label(progress_window, text="Starting reanalysis...")
        progress_label.pack()
        
        for i, idx in enumerate(selected_indices):
            row = results_df.iloc[idx]
            filename = row['Filename'] if 'Filename' in row else row['File']
            image_path = row['image_path'] if 'image_path' in row else os.path.join(row['Folder'] if 'Folder' in row else "", row['File'])
            
            progress_var.set(i)
            progress_label.config(text=f"Processing {i+1}/{len(selected_indices)}: {filename}")
            progress_window.update()
            
            # Ask for analysis mode
            choice = messagebox.askquestion(
                "Analysis Mode",
                f"Reanalyze {filename}\nUse automatic detection?",
                icon='question',
                parent=progress_window
            )
            
            try:
                if choice == 'yes':
                    contour, result = self.processor.enhanced_detection(image_path)
                    if contour is None or not result.success:
                        result = self.manual_tool.manual_analysis(image_path)
                else:
                    contour, result = self.manual_tool.manual_analysis(image_path)
                
                if contour is not None and isinstance(result, AnalysisResult) and result.success:
                    # Save processed image
                    processed_path = self.processor.save_processed_image(image_path, contour, result, output_folder)
                    
                    # Calculate scaled measurements
                    area_um2 = result.area * (scale ** 2)
                    length_um = result.major_axis_length * scale
                    width_um = result.minor_axis_length * scale
                    volume_mm3 = calculate_volume(length_um, width_um)
                    
                    # Update dataframe
                    results_df.at[idx, 'Mode'] = result.mode
                    results_df.at[idx, 'Area_um2'] = area_um2
                    results_df.at[idx, 'Length_um'] = length_um
                    results_df.at[idx, 'Width_um'] = width_um
                    results_df.at[idx, 'Volume_mm3'] = volume_mm3
                    results_df.at[idx, 'Roundness'] = result.roundness
                    results_df.at[idx, 'Solidity'] = result.solidity
                    results_df.at[idx, 'Aspect_Ratio'] = result.aspect_ratio
                    results_df.at[idx, 'Confidence'] = result.confidence_score
                    
                    # Update legacy columns if they exist
                    if 'Area (µm²)' in results_df.columns:
                        results_df.at[idx, 'Area (µm²)'] = area_um2
                    if 'Length (µm)' in results_df.columns:
                        results_df.at[idx, 'Length (µm)'] = length_um
                    if 'Width (µm)' in results_df.columns:
                        results_df.at[idx, 'Width (µm)'] = width_um
                    if 'Volume (mm³)' in results_df.columns:
                        results_df.at[idx, 'Volume (mm³)'] = volume_mm3
                    
            except Exception as e:
                logger.error(f"Error reanalyzing {filename}: {e}")
                messagebox.showerror("Error", f"Failed to reanalyze {filename}:\n{str(e)}")
        
        progress_window.destroy()
        messagebox.showinfo("Complete", f"Reanalyzed {len(selected_indices)} images")
        parent_window.destroy()
        self.show_reanalysis_table(results_df, scale, output_folder)
    
    def _export_to_excel(self, results_df):
        """Export results to Excel with multiple sheets"""
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if save_path:
                with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                    # Get unique folders
                    if 'Folder' in results_df.columns:
                        folders = results_df['Folder'].unique()
                        for folder in folders:
                            # Create safe sheet name
                            sheet_name = os.path.basename(folder)
                            sheet_name = re.sub(r'[\\/*?:\[\]]', '_', sheet_name)[:31]
                            
                            # Filter data for this folder
                            folder_df = results_df[results_df['Folder'] == folder]
                            
                            # Write to Excel
                            folder_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # Format worksheet
                            workbook = writer.book
                            worksheet = writer.sheets[sheet_name]
                            
                            # Header format
                            header_format = workbook.add_format({
                                'bold': True,
                                'text_wrap': True,
                                'valign': 'top',
                                'fg_color': '#4CAF50',
                                'font_color': 'white',
                                'border': 1
                            })
                            
                            # Apply header formatting
                            for col_num, value in enumerate(folder_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                            
                            # Auto-adjust column widths
                            for column in folder_df:
                                column_width = max(
                                    folder_df[column].astype(str).map(len).max(),
                                    len(column)
                                ) + 2
                                col_idx = folder_df.columns.get_loc(column)
                                worksheet.set_column(col_idx, col_idx, min(column_width, 25))
                    
                    # Also save a combined sheet
                    results_df.to_excel(writer, sheet_name="Combined_Results", index=False)
                
                messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
                webbrowser.open(save_path)
                
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")
    
    def _copy_to_clipboard(self, df: pd.DataFrame):
        """Copy results to clipboard"""
        try:
            # Select key columns
            cols = ['File', 'Area (µm²)', 'Length (µm)', 'Width (µm)', 'Volume (mm³)', 'Mode', 'Confidence']
            cols = [c for c in cols if c in df.columns]
            df_copy = df[cols].copy()
            
            # Format numbers
            for col in ['Area (µm²)', 'Length (µm)', 'Width (µm)', 'Volume (mm³)']:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            
            if 'Confidence' in df_copy.columns:
                df_copy['Confidence'] = df_copy['Confidence'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
            
            pyperclip.copy(df_copy.to_string(index=False))
            messagebox.showinfo("Clipboard", "Formatted results copied to clipboard")
            
        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            messagebox.showerror("Clipboard Error", f"Failed to copy:\n{str(e)}")

class SpheroidSizerApp:
    """Main application with enhanced stitching features"""
    def __init__(self):
        self.processor = EnhancedImageProcessor()
        self.manual_tool = EnhancedManualTool()
        self.reanalysis_table = None
        self.root = tk.Tk()
        self.root.title("Enhanced Spheroid Sizer with Advanced Stitching")
        self.setup_gui()
    
    def setup_gui(self):
        """Setup main GUI"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
    
        # Set window size to 90% of screen size (adjust as needed)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
    
        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
    
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="Enhanced Spheroid Sizer v6.0", 
                font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50').pack(pady=20)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Features list
        features_text = """✓ IMPROVED automatic detection with 5 different strategies
✓ Much more flexible parameters for various spheroid shapes
✓ Live contour preview during detection process
✓ Single file and folder batch processing
✓ Hybrid detection around manual points (press H)
✓ Real-time parameter adjustment
✓ Enhanced confidence scoring system
✓ NEW: Reanalysis table with thumbnails and selection
✓ NEW: Processed images saved with contour overlays"""
        
        tk.Label(main_frame, text=features_text, justify=tk.LEFT, 
                font=('Arial', 11), bg='white', fg='#34495e').pack(pady=20)
        
        # Processing mode selection
        mode_frame = tk.LabelFrame(main_frame, text="Processing Mode", font=('Arial', 12, 'bold'))
        mode_frame.pack(fill=tk.X, pady=10)
        
        btn_frame = tk.Frame(mode_frame)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Process Single File", command=self.process_single_file,
                 font=('Arial', 12), bg='#3498db', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Process Folder", command=self.process_folder,
                 font=('Arial', 12), bg='#2ecc71', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Open Results Table", command=self.open_reanalysis_table,
                 font=('Arial', 12), bg='#e67e22', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        
        # Enhanced settings frame
        settings_frame = tk.LabelFrame(main_frame, text="Enhanced Detection Settings", font=('Arial', 12, 'bold'))
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Parameter controls with better ranges
        params_frame = tk.Frame(settings_frame)
        params_frame.pack(pady=10)
        
        # Row 0
        tk.Label(params_frame, text="Min Area:").grid(row=0, column=0, sticky='e', padx=5)
        self.min_area_var = tk.IntVar(value=1000)
        tk.Scale(params_frame, from_=500, to=10000, orient=tk.HORIZONTAL, variable=self.min_area_var,
                command=self.update_params).grid(row=0, column=1, padx=5)
        
        tk.Label(params_frame, text="Max Area:").grid(row=0, column=2, sticky='e', padx=5)
        self.max_area_var = tk.IntVar(value=100000)
        tk.Scale(params_frame, from_=10000, to=5000000, orient=tk.HORIZONTAL, variable=self.max_area_var,
                command=self.update_params).grid(row=0, column=3, padx=5)
        
        # Row 1
        tk.Label(params_frame, text="Min Roundness:").grid(row=1, column=0, sticky='e', padx=1)
        self.min_roundness_var = tk.DoubleVar(value=0.15)
        tk.Scale(params_frame, from_=0.05, to=0.8, resolution=0.05, orient=tk.HORIZONTAL, 
                variable=self.min_roundness_var, command=self.update_params).grid(row=1, column=1, padx=5)
        
        tk.Label(params_frame, text="Min Solidity:").grid(row=1, column=2, sticky='e', padx=5)
        self.min_solidity_var = tk.DoubleVar(value=0.4)
        tk.Scale(params_frame, from_=0.2, to=0.9, resolution=0.1, orient=tk.HORIZONTAL, 
                variable=self.min_solidity_var, command=self.update_params).grid(row=1, column=3, padx=5)
        
        # Info text
        info_text = """Detection Strategy: Uses 5 different algorithms automatically
• Adaptive thresholding with multiple methods
• Otsu thresholding with morphological operations  
• Edge-based detection with gap filling
• Watershed segmentation for overlapping objects
• Multi-level thresholding with contour approximation

NEW FEATURES:
• Processed images saved with colored contours (-detected/-manual suffix)
• Reanalysis table shows thumbnails with selection checkboxes
• Click thumbnails to zoom, select multiple for batch reanalysis
• Support for processing multiple folders in batch mode"""
        
        tk.Label(settings_frame, text=info_text, justify=tk.LEFT, 
                font=('Arial', 9), fg='#7f8c8d').pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Enhanced detection with reanalysis capabilities")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
         # Add Donate button in the status bar or menu
        donate_frame = tk.Frame(self.root, bg='#f8f9fa')
        donate_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
        tk.Button(
            donate_frame,
            text="☕ Buy Me a Coffee",
            command=self.open_donate_link,
            bg='#ffdd00',
            fg='black',
            font=('Arial', 9, 'bold')
        ).pack(side=tk.RIGHT, padx=10)

    def open_donate_link(self):
        """Open donation URL in browser"""
        donate_url = "https://paypal.me/AGSpheroidsizer"
        webbrowser.open_new(donate_url)

    def open_reanalysis_table(self):
        """Open the reanalysis table from saved results"""
        try:
            # Ask user to select the results CSV file
            csv_path = filedialog.askopenfilename(
                title="Select Results CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        
            if not csv_path:
                return
            
            # Load the results
            results_df = pd.read_csv(csv_path)
        
            # Get the output folder (same as CSV file location)
            output_folder = os.path.join(os.path.dirname(csv_path), "processed_images")
        
            # Ask for scale factor
            scale = simpledialog.askfloat("Scale Factor", "Enter scale (micrometers per pixel):", 
                                    parent=self.root, minvalue=0.001)
            if scale is None:
                return
        
            # Initialize reanalysis table if needed
            if self.reanalysis_table is None:
                self.reanalysis_table = ReanalysisTable(self.root, self.processor, self.manual_tool)
            
            # Show the table
            self.reanalysis_table.show_reanalysis_table(results_df, scale, output_folder)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open reanalysis table: {str(e)}")
            logger.error(f"Error opening reanalysis table: {e}")
    
    def update_params(self, value=None):
        """Update processing parameters"""
        self.processor.min_area = self.min_area_var.get()
        self.processor.max_area = self.max_area_var.get()
        self.processor.min_roundness = self.min_roundness_var.get()
        self.processor.min_solidity = self.min_solidity_var.get()
        self.status_var.set(f"Updated: Area={self.processor.min_area}-{self.processor.max_area}, "
                           f"Roundness≥{self.processor.min_roundness:.2f}, Solidity≥{self.processor.min_solidity:.1f}")
    
    def process_single_file(self):
        """Process a single file with enhanced detection"""
        file_path = None
        
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )
        except Exception as e:
            logger.warning(f"File dialog failed: {e}")
        
        # If file dialog failed, ask for manual input
        if not file_path:
            file_path = simpledialog.askstring(
                "File Path", 
                "Enter the full path to your image file:\n(e.g., /home/runner/MyProject/image.tif)",
                parent=self.root
            )
        
        if not file_path:
            return
        
        # Validate file exists
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File does not exist: {file_path}")
            return
        
        self.status_var.set(f"Processing: {os.path.basename(file_path)}")
        self.root.update()
        
        # Get scale
        scale = simpledialog.askfloat("Scale Factor", "Enter scale (micrometers per pixel):", 
                                     parent=self.root, minvalue=0.001)
        if scale is None:
            return
        
        # Ask for processing mode
        mode_choice = messagebox.askyesnocancel("Processing Mode", 
                                               "Choose processing mode:\n"
                                               "Yes = Enhanced Automatic (5 strategies with preview)\n"
                                               "No = Manual/Hybrid\n"
                                               "Cancel = Cancel")
        
        if mode_choice is None:
            return
        
        try:
            if mode_choice:  # Enhanced Automatic with preview
                self._process_automatic_with_preview(file_path, scale)
            else:  # Manual/Hybrid
                self._process_manual_hybrid(file_path, scale)
                
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            logger.error(f"Single file processing error: {e}")
        
        self.status_var.set("Ready - Enhanced detection with reanalysis capabilities")
    
    def _process_automatic_with_preview(self, file_path: str, scale: float):
        """Process with enhanced automatic detection and live preview"""
        preview = LivePreviewWindow("Enhanced Automatic Detection - 5 Strategies")
        preview.start()
        
        def preview_callback(img, status):
            preview.update(img, status)
        
        try:
            contour, result = self.processor.enhanced_detection(file_path, preview_callback)
            preview.close()
            
            if contour is not None and isinstance(result, AnalysisResult) and result.success:
                # Create output folder and save processed image
                output_folder = os.path.join(os.path.dirname(file_path), "processed_images")
                processed_path = self.processor.save_processed_image(file_path, contour, result, output_folder)
                
                self._show_results(file_path, contour, result, scale, processed_path)
            else:
                error_msg = result if isinstance(result, str) else "All 5 detection strategies failed"
                choice = messagebox.askyesno("Detection Failed", 
                                           f"Enhanced automatic detection failed: {error_msg}\n\n"
                                           "This image may have:\n"
                                           "• Very low contrast spheroids\n"
                                           "• Unusual shapes or artifacts\n"
                                           "• Background interference\n\n"
                                           "Would you like to try manual/hybrid mode?")
                if choice:
                    self._process_manual_hybrid(file_path, scale)
        finally:
            preview.close()
    
    def _process_manual_hybrid(self, file_path: str, scale: float):
        """Process with manual/hybrid mode"""
        messagebox.showinfo("Manual/Hybrid Mode", 
                           "Manual/Hybrid Mode Instructions:\n\n"
                           "• Left-click to add points around spheroid\n"
                           "• Right-click to remove last point\n"
                           "• Press 'H' for HYBRID detection around points\n"
                           "  (tries multiple methods in the marked area)\n"
                           "• Press Enter when finished (min 3 points)\n"
                           "• Press 'Q' to cancel\n\n"
                           "TIP: For hybrid mode, just mark 2-3 points near\n"
                           "the spheroid and press 'H' for automatic detection!")
        
        contour, result = self.manual_tool.manual_analysis(file_path)
        
        if contour is not None and isinstance(result, AnalysisResult) and result.success:
            # Create output folder and save processed image
            output_folder = os.path.join(os.path.dirname(file_path), "processed_images")
            processed_path = self.processor.save_processed_image(file_path, contour, result, output_folder)
            
            self._show_results(file_path, contour, result, scale, processed_path)
        else:
            error_msg = result if isinstance(result, str) else "Analysis cancelled"
            messagebox.showwarning("Analysis Failed", error_msg)
    
    def _show_results(self, file_path: str, contour, result: AnalysisResult, scale: float, processed_path: str = None):
        """Show analysis results with confidence indication"""
        # Calculate scaled measurements
        area_um2 = result.area * (scale ** 2)
        length_um = result.major_axis_length * scale
        width_um = result.minor_axis_length * scale
        
        # Estimate volume (assuming prolate spheroid)
        volume_mm3 = calculate_volume(length_um, width_um)
        
        # Confidence interpretation
        if result.confidence_score >= 0.8:
            confidence_text = "EXCELLENT (Very reliable)"
        elif result.confidence_score >= 0.6:
            confidence_text = "GOOD (Reliable)"
        elif result.confidence_score >= 0.4:
            confidence_text = "FAIR (Check manually)"
        else:
            confidence_text = "LOW (Manual verification recommended)"
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Enhanced Analysis Results")
        results_window.geometry("550x500")
        results_window.transient(self.root)
        
        # Results text
        results_text = f"""File: {os.path.basename(file_path)}
Detection Mode: {result.mode.upper()}
Confidence: {result.confidence_score:.3f} - {confidence_text}

MEASUREMENTS:
Area: {area_um2:.2f} µm²
Length (Major axis): {length_um:.2f} µm
Width (Minor axis): {width_um:.2f} µm
Estimated Volume: {volume_mm3:.6f} mm³

SHAPE METRICS:
Roundness: {result.roundness:.3f} (1.0 = perfect circle)
Solidity: {result.solidity:.3f} (convexity measure)
Aspect Ratio: {result.aspect_ratio:.2f} (length/width)
Perimeter: {result.perimeter:.1f} pixels

PROCESSING INFO:
Scale Factor: {scale:.3f} µm/pixel
Detection Strategy: Enhanced multi-algorithm approach
Total Pixels: {result.area:.0f}
Processed Image: {os.path.basename(processed_path) if processed_path else 'Not saved'}

QUALITY INDICATORS:
• Confidence Score: {result.confidence_score:.3f}/1.000
• Shape Quality: {'Excellent' if result.roundness > 0.6 else 'Good' if result.roundness > 0.4 else 'Fair'}
• Size Category: {'Large' if area_um2 > 10000 else 'Medium' if area_um2 > 2000 else 'Small'}"""
        
        text_widget = tk.Text(results_window, wrap=tk.WORD, font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.INSERT, results_text)
        text_widget.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = tk.Frame(results_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="Copy Results", 
                 command=lambda: self._copy_results(results_text)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Results", 
                 command=lambda: self._save_results(file_path, results_text)).pack(side=tk.LEFT, padx=5)
        
        # Show processed image button
        if processed_path and os.path.exists(processed_path):
            tk.Button(btn_frame, text="View Processed Image", 
                     command=lambda: self._show_processed_image(processed_path)).pack(side=tk.LEFT, padx=5)
        
        # Reprocess button
        tk.Button(btn_frame, text="Try Different Mode", 
                 command=lambda: [results_window.destroy(), self.process_single_file()]).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Close", 
                 command=results_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _show_processed_image(self, image_path):
        """Show the processed image with contours"""
        try:
            img_window = tk.Toplevel(self.root)
            img_window.title(f"Processed Image - {os.path.basename(image_path)}")
            img_window.geometry("600x600")
            
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((580, 580), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_pil)
                
                label = tk.Label(img_window, image=photo)
                label.pack(padx=10, pady=10)
                
                # Keep reference
                label.image = photo
        except Exception as e:
            logger.error(f"Error showing processed image: {e}")
    
    def _copy_results(self, text: str):
        """Copy results to clipboard"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Results copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
    
    def _save_results(self, image_path: str, results_text: str):
        """Save results to file"""
        try:
            save_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(results_text)
                messagebox.showinfo("Saved", f"Results saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def process_folder(self):
        """Process folder of images with enhanced detection"""
        # Try file dialog first, with fallback to manual input
        folder_path = None
        
        try:
            folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        except Exception as e:
            logger.warning(f"File dialog failed: {e}")
        
        # If file dialog failed or user cancelled, ask for manual input
        if not folder_path:
            folder_path = simpledialog.askstring(
                "Folder Path", 
                "Enter the full path to your image folder:\n(e.g., /home/runner/MyProject/images)",
                parent=self.root
            )
        
        if not folder_path:
            messagebox.showwarning("No Selection", "No folder selected.")
            return
        
        # Validate folder exists
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Folder does not exist: {folder_path}")
            return
        
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", f"Path is not a directory: {folder_path}")
            return
        
        self.status_var.set(f"Processing folder: {folder_path}")
        self.root.update()
        
        # Get scale
        scale = simpledialog.askfloat("Scale Factor", "Enter scale (micrometers per pixel):", 
                                     parent=self.root, minvalue=0.001)
        if scale is None:
            return
        
        # Process images in folder
        try:
            results_df = self._process_folder_batch(folder_path, scale)
            if results_df is not None and not results_df.empty:
                # Show reanalysis table
                if self.reanalysis_table is None:
                    self.reanalysis_table = ReanalysisTable(self.root, self.processor, self.manual_tool)
                
                output_folder = os.path.join(folder_path, "processed_images")
                self.reanalysis_table.show_reanalysis_table(results_df, scale, output_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Folder processing failed: {str(e)}")
            logger.error(f"Folder processing error: {e}")
        
        self.status_var.set("Ready - Enhanced detection with reanalysis capabilities")
    
     
    def _process_folder_batch(self, folder_path: str, scale: float):
        """Process all images in a single folder"""
        try:
            # Get image files
            image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                messagebox.showwarning("No Images", "No image files found in the selected folder.")
                return None
            
            messagebox.showinfo("Enhanced Folder Processing", 
                               f"Found {len(image_files)} images.\n\n"
                               "Enhanced Detection Process:\n"
                               "• Each image will be processed with 5 different strategies\n"
                               "• Live preview shows detection progress\n"
                               "• Failed detections offer manual correction\n"
                               "• Results include confidence scores\n"
                               "• Processed images saved with contour overlays\n"
                               "• Reanalysis table will be shown at the end\n\n"
                               "This may take longer but provides much better results!")
            
            # Create output folder
            output_folder = os.path.join(folder_path, "processed_images")
            os.makedirs(output_folder, exist_ok=True)
            
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Enhanced Batch Processing")
            progress_window.geometry("600x350")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(image_files))
            progress_bar.pack(padx=20, pady=20, fill=tk.X)
            
            status_label = tk.Label(progress_window, text="Starting enhanced batch processing...")
            status_label.pack(pady=10)
            
            results_text = tk.Text(progress_window, height=10, width=80)
            results_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
            
            cancel_var = tk.BooleanVar(value=False)
            cancel_btn = tk.Button(progress_window, text="Cancel", 
                                  command=lambda: cancel_var.set(True))
            cancel_btn.pack(pady=5)
            
            # Process images
            results = []
            successful = 0
            failed = 0
            manual_count = 0
            
            for i, filename in enumerate(image_files):
                if cancel_var.get():
                    break
                
                progress_var.set(i)
                status_label.config(text=f"Processing {i+1}/{len(image_files)}: {filename}")
                progress_window.update()
                
                image_path = os.path.join(folder_path, filename)
                
                try:
                    # Try enhanced automatic detection first
                    preview = LivePreviewWindow(f"Processing {filename}")
                    preview.start()
                    
                    def preview_callback(img, status):
                        preview.update(img, f"{filename}: {status}")
                    
                    contour, result = self.processor.enhanced_detection(image_path, preview_callback)
                    preview.close()
                    
                    # If automatic fails or user wants manual, switch to manual
                    if contour is None or not isinstance(result, AnalysisResult) or not result.success:
                        choice = messagebox.askyesno("Manual Mode", 
                                                   f"Automatic detection failed for {filename}\n"
                                                   f"Reason: {result if isinstance(result, str) else 'All strategies failed'}\n\n"
                                                   "Would you like to try manual mode?",
                                                   parent=progress_window)
                        if choice:
                            manual_count += 1
                            messagebox.showinfo("Manual Mode", 
                                              f"Manual mode for {filename}\n"
                                              "Click points around spheroid, press H for hybrid, Enter when done",
                                              parent=progress_window)
                            contour, result = self.manual_tool.manual_analysis(image_path)
                    
                    if contour is not None and isinstance(result, AnalysisResult) and result.success:
                        # Save processed image
                        processed_path = self.processor.save_processed_image(image_path, contour, result, output_folder)
                        
                        # Calculate scaled measurements
                        area_um2 = result.area * (scale ** 2)
                        length_um = result.major_axis_length * scale
                        width_um = result.minor_axis_length * scale
                        volume_mm3 = calculate_volume(length_um, width_um)
                        
                        results.append({
                            'Filename': filename,
                            'Mode': result.mode,
                            'Area_um2': area_um2,
                            'Length_um': length_um,
                            'Width_um': width_um,
                            'Volume_mm3': volume_mm3,
                            'Roundness': result.roundness,
                            'Solidity': result.solidity,
                            'Aspect_Ratio': result.aspect_ratio,
                            'Confidence': result.confidence_score,
                            'image_path': image_path,
                            'processed_path': processed_path
                        })
                        
                        successful += 1
                        result_line = f"{filename}: {result.mode} - Area: {area_um2:.1f}µm², Volume: {volume_mm3:.6f}mm³, Conf: {result.confidence_score:.2f}\n"
                    else:
                        failed += 1
                        result_line = f"{filename}: Failed\n"
                    
                    results_text.insert(tk.END, result_line)
                    results_text.see(tk.END)
                    progress_window.update()
                
                except Exception as e:
                    failed += 1
                    error_line = f"{filename}: Error - {str(e)}\n"
                    results_text.insert(tk.END, error_line)
                    logger.error(f"Error processing {filename}: {e}")
                    continue
            
            progress_window.destroy()
            
            if not results:
                return None
            
            # Create dataframe
            results_df = pd.DataFrame(results)
            
            # Show completion message
            completion_msg = f"Enhanced processing finished!\n\n" \
                            f"Successful: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)\n" \
                            f"Automatic detections: {successful - manual_count}\n" \
                            f"Manual corrections: {manual_count}\n" \
                            f"Failed: {failed}\n\n" \
                            f"✓ Results saved with confidence scores\n" \
                            f"✓ Processed images saved with contour overlays\n" \
                            f"✓ Reanalysis table will open next"
            
            messagebox.showinfo("Enhanced Batch Complete", completion_msg)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            return None
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    app = SpheroidSizerApp()
    app.run()

if __name__ == "__main__":
    main()
