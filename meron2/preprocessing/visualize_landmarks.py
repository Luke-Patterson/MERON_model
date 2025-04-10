import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Dict, Optional, List, Tuple
from facial_landmarks_processor import FacialLandmarksProcessor

class LandmarkVisualizer:
    """Visualize facial landmarks and extracted features."""
    
    def __init__(self, processor: FacialLandmarksProcessor):
        self.processor = processor
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 2) -> np.ndarray:
        """
        Draw facial landmarks on the image.
        
        Args:
            image: RGB image
            landmarks: Array of shape (68, 2) containing landmark coordinates
            color: BGR color tuple for drawing
            thickness: Line thickness
            
        Returns:
            Image with landmarks drawn
        """
        img = image.copy()
        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(img, (x, y), 1, color, thickness)
        return img
    
    def draw_facial_regions(self, image: np.ndarray, landmarks: np.ndarray,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          alpha: float = 0.3) -> np.ndarray:
        """
        Draw facial regions using landmarks.
        
        Args:
            image: RGB image
            landmarks: Array of shape (68, 2) containing landmark coordinates
            color: BGR color tuple for drawing
            alpha: Transparency of the overlay
            
        Returns:
            Image with facial regions drawn
        """
        img = image.copy()
        overlay = img.copy()
        
        # Draw each facial region
        regions = {
            'jaw': landmarks[self.processor.FACIAL_REGIONS['jaw']],
            'left_eyebrow': landmarks[self.processor.FACIAL_REGIONS['left_eyebrow']],
            'right_eyebrow': landmarks[self.processor.FACIAL_REGIONS['right_eyebrow']],
            'nose_bridge': landmarks[self.processor.FACIAL_REGIONS['nose_bridge']],
            'nose_tip': landmarks[self.processor.FACIAL_REGIONS['nose_tip']],
            'left_eye': landmarks[self.processor.FACIAL_REGIONS['left_eye']],
            'right_eye': landmarks[self.processor.FACIAL_REGIONS['right_eye']],
            'outer_mouth': landmarks[self.processor.FACIAL_REGIONS['outer_mouth']],
            'inner_mouth': landmarks[self.processor.FACIAL_REGIONS['inner_mouth']]
        }
        
        for region_name, points in regions.items():
            points = points.astype(np.int32)
            cv2.fillPoly(overlay, [points], color)
        
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
    
    def visualize_features(self, image: np.ndarray, processed_data: Dict,
                         show_landmarks: bool = True,
                         show_regions: bool = True,
                         show_measurements: bool = True) -> None:
        """
        Create a comprehensive visualization of facial features.
        
        Args:
            image: Original RGB image
            processed_data: Output from FacialLandmarksProcessor.process_image()
            show_landmarks: Whether to show landmark points
            show_regions: Whether to show facial regions
            show_measurements: Whether to show feature measurements
        """
        landmarks = processed_data['landmarks']
        features = processed_data['geometric_features']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Original image with landmarks
        if show_landmarks:
            ax1 = fig.add_subplot(221)
            img_landmarks = self.draw_landmarks(image, landmarks)
            ax1.imshow(cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2RGB))
            ax1.set_title('Facial Landmarks')
            ax1.axis('off')
        
        # Facial regions
        if show_regions:
            ax2 = fig.add_subplot(222)
            img_regions = self.draw_facial_regions(image, landmarks)
            ax2.imshow(cv2.cvtColor(img_regions, cv2.COLOR_BGR2RGB))
            ax2.set_title('Facial Regions')
            ax2.axis('off')
        
        # Feature measurements
        if show_measurements:
            ax3 = fig.add_subplot(223)
            # Select key features to display
            key_features = {
                'cheek_symmetry': 'Cheek Symmetry',
                'temple_width': 'Temple Width',
                'face_ratio': 'Face Ratio',
                'eye_symmetry': 'Eye Symmetry',
                'mouth_ratio': 'Mouth Ratio',
                'jaw_angle': 'Jaw Angle',
                'forehead_angle': 'Forehead Angle'
            }
            
            values = [features[k] for k in key_features.keys()]
            labels = list(key_features.values())
            
            ax3.bar(labels, values)
            ax3.set_title('Key Feature Measurements')
            plt.xticks(rotation=45, ha='right')
            
            # Texture features if available
            if 'left_cheek_texture_mean' in features:
                ax4 = fig.add_subplot(224)
                texture_features = {
                    'left_cheek_texture_mean': 'Left Cheek Mean',
                    'right_cheek_texture_mean': 'Right Cheek Mean',
                    'left_eye_texture_mean': 'Left Eye Mean',
                    'right_eye_texture_mean': 'Right Eye Mean'
                }
                
                values = [features[k] for k in texture_features.keys()]
                labels = list(texture_features.values())
                
                ax4.bar(labels, values)
                ax4.set_title('Texture Features')
                plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_landmark_detection(self, image_path: str) -> None:
        """
        Visualize the landmark detection process for a single image.
        
        Args:
            image_path: Path to the image file
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Process image
        processed_data = self.processor.process_image(img)
        if processed_data is None:
            raise ValueError("No face detected in image")
        
        # Create visualization
        self.visualize_features(img, processed_data)
    
    def create_feature_report(self, processed_data: Dict) -> str:
        """
        Create a text report of the extracted features.
        
        Args:
            processed_data: Output from FacialLandmarksProcessor.process_image()
            
        Returns:
            Formatted string containing feature information
        """
        features = processed_data['geometric_features']
        
        report = "Facial Feature Analysis Report\n"
        report += "=" * 30 + "\n\n"
        
        # Basic measurements
        report += "Basic Measurements:\n"
        report += "-" * 20 + "\n"
        basic_features = ['cheek_symmetry', 'temple_width', 'face_ratio', 
                         'eye_symmetry', 'mouth_ratio']
        for feat in basic_features:
            report += f"{feat}: {features[feat]:.4f}\n"
        
        # Area measurements
        report += "\nArea Measurements:\n"
        report += "-" * 20 + "\n"
        area_features = ['face_area', 'left_cheek_area', 'right_cheek_area',
                        'left_eye_area', 'right_eye_area']
        for feat in area_features:
            report += f"{feat}: {features[feat]:.4f}\n"
        
        # Angular measurements
        report += "\nAngular Measurements:\n"
        report += "-" * 20 + "\n"
        angle_features = ['jaw_angle', 'forehead_angle', 'left_jaw_angle',
                         'right_jaw_angle']
        for feat in angle_features:
            report += f"{feat}: {features[feat]:.4f} radians\n"
        
        # Texture features (if available)
        if 'left_cheek_texture_mean' in features:
            report += "\nTexture Features:\n"
            report += "-" * 20 + "\n"
            texture_features = ['left_cheek_texture_mean', 'right_cheek_texture_mean',
                              'left_eye_texture_mean', 'right_eye_texture_mean']
            for feat in texture_features:
                report += f"{feat}: {features[feat]:.4f}\n"
        
        return report 