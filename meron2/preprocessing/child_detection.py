import cv2
import numpy as np
import os
from pathlib import Path

class ChildDetector:
    def __init__(self, model_dir=None):
        """
        Initialize the child detector with optional model directory
        Args:
            model_dir: Path to directory containing model files. If None, uses default preprocess_models directory
        """
        if model_dir is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            model_dir = script_dir / 'preprocess_models'
        else:
            model_dir = Path(model_dir)
            
        # Load the pre-trained model for person detection
        self.model = cv2.dnn.readNetFromCaffe(
            str(model_dir / 'deploy.prototxt'),
            str(model_dir / 'res10_300x300_ssd_iter_140000.caffemodel')
        )
        
    def detect_child(self, image):
        """
        Detect child in the image and return bounding box coordinates
        """
        height, width = image.shape[:2]
        
        # Preprocess image for the model
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), [104, 117, 123], False, False
        )
        
        # Run detection
        self.model.setInput(blob)
        detections = self.model.forward()
        
        # Find the largest detection (likely the main subject)
        max_area = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate area
                area = (endX - startX) * (endY - startY)
                if area > max_area:
                    max_area = area
                    best_box = (startX, startY, endX, endY)
        
        return best_box
    
    def crop_to_child(self, image, box):
        """
        Crop image to the detected child region with some padding
        """
        if box is None:
            return image
            
        startX, startY, endX, endY = box
        
        # Add padding (20% of the box size)
        height, width = image.shape[:2]
        pad_x = int((endX - startX) * 0.2)
        pad_y = int((endY - startY) * 0.2)
        
        # Ensure padding doesn't go outside image boundaries
        startX = max(0, startX - pad_x)
        startY = max(0, startY - pad_y)
        endX = min(width, endX + pad_x)
        endY = min(height, endY + pad_y)
        
        return image[startY:endY, startX:endX]

def process_image(input_path, output_dir, model_dir=None):
    """
    Process a single image: detect child and crop
    Args:
        input_path: Path to input image
        output_dir: Directory to save processed images
        model_dir: Optional path to directory containing model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error reading image: {input_path}")
        return False
    
    # Get filename without extension and extension
    filename = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    
    # Initialize detector
    detector = ChildDetector(model_dir)
    
    # Detect child and crop
    box = detector.detect_child(image)
    cropped_image = detector.crop_to_child(image, box)
    
    # Save cropped image
    cropped_path = os.path.join(output_dir, f"{filename}{ext}")
    cv2.imwrite(cropped_path, cropped_image)
    
    return True

def process_directory(input_dir, output_dir, model_dir=None):
    """
    Process all images in a directory
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        model_dir: Optional path to directory containing model files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    for img_path in input_path.glob('**/*.jpg'):
        print(f"Processing: {img_path}")
        success = process_image(str(img_path), str(output_path), model_dir)
        if not success:
            print(f"Failed to process: {img_path}") 