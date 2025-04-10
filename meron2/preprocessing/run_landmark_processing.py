import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from facial_landmarks_processor import FacialLandmarksProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('landmark_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_dataset(image_dir: Path, output_dir: Path):
    """
    Process all images in directory and save extracted features.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save processed features
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = FacialLandmarksProcessor()
    
    # Lists to store results
    processed_data = []
    failed_images = []
    
    # Process all images
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
    logger.info(f"Found {len(image_files)} images to process")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                failed_images.append(str(img_path))
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process image
            result = processor.process_image(img)
            if result is None:
                logger.warning(f"Failed to detect face in: {img_path}")
                failed_images.append(str(img_path))
                continue
                
            # Create feature vector
            features = processor.create_feature_vector(result)
            
            # Store results
            processed_data.append({
                'photo_id': img_path.stem,
                'features': features,
                'geometric_features': result['geometric_features']
            })
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            failed_images.append(str(img_path))
    
    # Save results
    if processed_data:
        # Save main feature vectors
        feature_arrays = np.stack([d['features'] for d in processed_data])
        photo_ids = [d['photo_id'] for d in processed_data]
        
        feature_df = pd.DataFrame(
            feature_arrays,
            index=photo_ids,
            columns=[f'landmark_feat_{i}' for i in range(feature_arrays.shape[1])]
        )
        feature_df.index.name = 'photo_id'
        feature_df.to_csv(output_dir / 'landmark_features.csv')
        
        # Save geometric features separately
        geometric_df = pd.DataFrame([
            {**{'photo_id': d['photo_id']}, **d['geometric_features']}
            for d in processed_data
        ])
        geometric_df.set_index('photo_id', inplace=True)
        geometric_df.to_csv(output_dir / 'geometric_features.csv')
        
        # Save list of failed images
        if failed_images:
            with open(output_dir / 'failed_images.txt', 'w') as f:
                f.write('\n'.join(failed_images))
        
        logger.info(f"Successfully processed {len(processed_data)} images")
        logger.info(f"Failed to process {len(failed_images)} images")
        
    else:
        logger.error("No images were successfully processed")

if __name__ == '__main__':
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    image_dir = base_dir / 'data' / 'cropped_pictures'
    output_dir = base_dir / 'data' / 'processed' / 'landmarks'
    
    # Run processing
    process_dataset(image_dir, output_dir) 