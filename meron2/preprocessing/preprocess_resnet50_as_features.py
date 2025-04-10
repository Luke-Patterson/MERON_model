import os
import pandas as pd
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MERONPreprocessor:
    def __init__(self):
        # Paths
        self.data_dir = '../../data/linkage_data'  # Updated path
        self.image_dir = '../../data/cropped_pictures'  # Updated path
        self.output_dir = '../../data/processed'  # Updated path
        
        # Print absolute paths for debugging
        logging.info(f"Data directory (absolute): {os.path.abspath(self.data_dir)}")
        logging.info(f"Image directory (absolute): {os.path.abspath(self.image_dir)}")
        logging.info(f"Output directory (absolute): {os.path.abspath(self.output_dir)}")
        
        # Initialize ResNet50 model
        self.model = None  # Will be loaded when needed
        self.target_size = (224, 224)  # ResNet50 expected input size
        
    def load_metadata(self):
        """Load and combine metadata from all county Excel files"""
        all_data = []
        excel_files = ['Turkana.xlsx', 'Marsabit.xlsx', 'Isiolo.xlsx', 'Tana River.xlsx']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        for file in excel_files:
            file_path = os.path.join(self.data_dir, file)
            logging.info(f"Attempting to load: {file_path}")
            logging.info(f"File exists: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
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
    
    def initialize_model(self):
        """Initialize ResNet50 model for feature extraction"""
        logging.info("Initializing ResNet50 model")
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image for ResNet50"""
        try:
            img = image.load_img(image_path, target_size=self.target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None
        
    def process_images(self, metadata_df):
        """Extract ResNet50 features from images"""
        if self.model is None:
            self.initialize_model()
            
        features_list = []
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
                image_path = os.path.join(self.image_dir, photo_id)
                img_data = self.load_and_preprocess_image(image_path)
                
                if img_data is not None:
                    # Extract features
                    features = self.model.predict(img_data, verbose=0)
                    features_list.append(features.flatten())
                    processed_photo_ids.append(row['photo_id'])
            else:
                logging.warning(f"Image not found for photo_id: {photo_id}")
        
        # Create features DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df.insert(0, 'photo_id', processed_photo_ids)
            logging.info(f"Successfully processed {len(features_list)} images")
            return features_df
        else:
            raise ValueError("No images were successfully processed")
    
    def calculate_wfh_zscore(self, metadata_df):
        """Calculate Weight-for-Height z-scores"""
        # To be implemented - will use WHO standards
        pass
    
    def save_features(self, features_df, output_file):
        """Save extracted features with metadata"""
        output_path = os.path.join(self.output_dir, output_file)
        features_df.to_csv(output_path, index=False)
        logging.info(f"Saved features to {output_path}")

def main():
    try:
        logging.info("Starting MERON preprocessing pipeline")
        processor = MERONPreprocessor()
        
        # Load metadata
        metadata = processor.load_metadata()
        logging.info("Metadata loaded successfully")
        
        # Process images and extract features
        features_df = processor.process_images(metadata)
        logging.info("Feature extraction completed")
        
        # Save features
        processor.save_features(features_df, 'resnet50_features.csv')
        
        # Display statistics
        logging.info(f"Metadata shape: {metadata.shape}")
        logging.info(f"Features shape: {features_df.shape}")
        logging.info(f"Number of processed images: {len(features_df)}")
            
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main() 