import os
from pathlib import Path
from child_detection import process_image

def main():
    # Get base directory from environment variable, fallback to parent of preprocessing folder
    base_dir = Path(os.getenv('MERON_BASE_DIR', Path(__file__).parent.parent))
    raw_pics_dir = base_dir / 'data/raw_pictures'
    output_dir = base_dir / 'data/cropped_pictures'
    
    # Get model directory from environment variable, fallback to default
    model_dir = os.getenv('MERON_MODEL_DIR')
    if model_dir:
        print(f"Using model directory: {model_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images recursively from all subfolders
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:  # Support multiple image formats
        image_files.extend(list(raw_pics_dir.rglob(ext)))
    
    if not image_files:
        print(f"No images found in {raw_pics_dir} or its subfolders")
        return
        
    print(f"Found {len(image_files)} images in {raw_pics_dir} and its subfolders")
    
    # Process all images
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path}")
        success = process_image(str(img_path), str(output_dir), model_dir)
        if not success:
            print(f"Failed to process: {img_path}")

if __name__ == "__main__":
    main() 