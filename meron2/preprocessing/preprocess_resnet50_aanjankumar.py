import os
import pandas as pd
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging
from tqdm import tqdm
import cv2
import dlib
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
LINKAGE_DIR = DATA_DIR / 'linkage_data'
IMAGE_DIR = DATA_DIR / 'cropped_pictures'
OUTPUT_DIR = DATA_DIR / 'processed'
LANDMARK_FILE = DATA_DIR / 'shape_predictor_68_face_landmarks.dat'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)

class MERONPreprocessorAanjan:
    def __init__(self):
        self.data_dir = LINKAGE_DIR
        self.image_dir = IMAGE_DIR
        self.output_dir = OUTPUT_DIR
        self.landmark_file = LANDMARK_FILE
        
        # Log absolute paths for debugging
        logging.info(f"Data directory (absolute): {self.data_dir.absolute()}")
        logging.info(f"Image directory (absolute): {self.image_dir.absolute()}")
        logging.info(f"Output directory (absolute): {self.output_dir.absolute()}")
        
        # Initialize models
        self.model = None
        self.detector = None
        self.predictor = None
        self.target_size = (224, 224)  # Standard ResNet input size
        
    def load_metadata(self):
        """Load and combine metadata from all county Excel files"""
        all_data = []
        excel_files = ['Turkana.xlsx', 'Marsabit.xlsx', 'Isiolo.xlsx', 'Tana River.xlsx']
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for file in excel_files:
            file_path = self.data_dir / file
            logging.info(f"Attempting to load: {file_path}")
            logging.info(f"File exists: {file_path.exists()}")
            
            if file_path.exists():
                logging.info(f"Loading metadata from {file}")
                try:
                    df = pd.read_excel(file_path)
                    logging.info(f"Successfully loaded {file} with {len(df)} rows")
                    all_data.append(df)
                except Exception as e:
                    logging.error(f"Error loading {file}: {str(e)}")
            else:
                logging.warning(f"File not found: {file_path}")
        
        if not all_data:
            raise FileNotFoundError("No metadata files found")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total records loaded: {len(combined_data)}")
        return combined_data
    
    def initialize_models(self):
        """Initialize ResNet50 and facial landmark detection models"""
        logging.info("Initializing models")
        # Initialize ResNet50
        self.model = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        # Initialize dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(self.landmark_file))
    
    def detect_landmarks(self, img):
        """Detect facial landmarks using dlib"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            if len(faces) == 0:
                logging.warning("No face detected in image")
                return None
                
            # Get landmarks for the first face
            landmarks = self.predictor(gray, faces[0])
            
            # Extract landmark coordinates
            points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                points.append([x, y])
                
            return np.array(points)
            
        except Exception as e:
            logging.error(f"Error detecting landmarks: {str(e)}")
            return None
    
    def preprocess_image(self, image_path):
        """Preprocess image according to paper methodology"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect facial landmarks
            landmarks = self.detect_landmarks(img)
            if landmarks is None:
                return None
            
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            # Apply ResNet50 preprocessing
            img = preprocess_input(img)
            
            # Expand dimensions for model input
            img = np.expand_dims(img, axis=0)
            
            return img, landmarks
            
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_features(self, img_data):
        """Extract features using ResNet50"""
        if img_data is None:
            return None
        
        try:
            # Extract features
            features = self.model.predict(img_data, verbose=0)
            return features.flatten()
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None
    
    def process_images(self, metadata_df):
        """Process images and extract features according to paper methodology"""
        if self.model is None:
            self.initialize_models()
        
        features_list = []
        landmarks_list = []
        processed_photo_ids = []
        
        # Get list of available image files
        available_images = set(os.listdir(self.image_dir))
        logging.info(f"Found {len(available_images)} images in directory")
        
        # Process each image that has metadata
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing images"):
            photo_id = str(row['photo_id'])
            if not photo_id.endswith('.jpg'):
                photo_id += '.jpg'
            
            if photo_id in available_images:
                image_path = self.image_dir / photo_id
                
                # Preprocess image and get landmarks
                result = self.preprocess_image(image_path)
                
                if result is not None:
                    img_data, landmarks = result
                    
                    # Extract features
                    features = self.extract_features(img_data)
                    
                    if features is not None:
                        features_list.append(features)
                        landmarks_list.append(landmarks)
                        processed_photo_ids.append(row['photo_id'])
            else:
                logging.warning(f"Image not found for photo_id: {photo_id}")
        
        # Create features DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df.insert(0, 'photo_id', processed_photo_ids)
            
            # Save landmarks separately
            landmarks_df = pd.DataFrame({
                'photo_id': processed_photo_ids,
                'landmarks': landmarks_list
            })
            
            logging.info(f"Successfully processed {len(features_list)} images")
            return features_df, landmarks_df
        else:
            raise ValueError("No images were successfully processed")
    
    def save_features(self, features_df, landmarks_df, output_file):
        """Save extracted features and landmarks with metadata"""
        # Save features
        features_path = self.output_dir / output_file
        features_df.to_csv(features_path, index=False)
        logging.info(f"Saved features to {features_path}")
        
        # Save landmarks
        landmarks_path = self.output_dir / output_file.replace('.csv', '_landmarks.csv')
        landmarks_df.to_csv(landmarks_path, index=False)
        logging.info(f"Saved landmarks to {landmarks_path}")

def main():
    try:
        logging.info("Starting MERON preprocessing pipeline (Aanjan Methodology)")
        processor = MERONPreprocessorAanjan()
        
        # Load metadata
        metadata = processor.load_metadata()
        logging.info("Metadata loaded successfully")
        
        # Process images and extract features
        features_df, landmarks_df = processor.process_images(metadata)
        logging.info("Feature extraction completed")
        
        # Save features and landmarks
        processor.save_features(features_df, landmarks_df, 'resnet50_features_aanjankumar.csv')
        
        # Display statistics
        logging.info(f"Metadata shape: {metadata.shape}")
        logging.info(f"Features shape: {features_df.shape}")
        logging.info(f"Number of processed images: {len(features_df)}")
            
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 