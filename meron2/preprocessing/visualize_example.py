import os
import cv2
import argparse
from facial_landmarks_processor import FacialLandmarksProcessor
from visualize_landmarks import LandmarkVisualizer

def main():
    parser = argparse.ArgumentParser(description='Visualize facial landmarks and features')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to dlib facial landmark model')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Directory to save output visualizations')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor and visualizer
    processor = FacialLandmarksProcessor(landmark_model_path=args.model_path)
    visualizer = LandmarkVisualizer(processor)
    
    # Process and visualize the image
    try:
        # Read image
        img = cv2.imread(args.image_path)
        if img is None:
            raise ValueError(f"Could not read image: {args.image_path}")
        
        # Process image
        processed_data = processor.process_image(img)
        if processed_data is None:
            raise ValueError("No face detected in image")
        
        # Create visualizations
        print("Creating visualizations...")
        
        # 1. Draw landmarks
        img_landmarks = visualizer.draw_landmarks(img, processed_data['landmarks'])
        cv2.imwrite(os.path.join(args.output_dir, 'landmarks.jpg'), img_landmarks)
        
        # 2. Draw facial regions
        img_regions = visualizer.draw_facial_regions(img, processed_data['landmarks'])
        cv2.imwrite(os.path.join(args.output_dir, 'regions.jpg'), img_regions)
        
        # 3. Create feature report
        report = visualizer.create_feature_report(processed_data)
        with open(os.path.join(args.output_dir, 'feature_report.txt'), 'w') as f:
            f.write(report)
        
        # 4. Show interactive visualization
        print("\nFeature Report:")
        print(report)
        print("\nShowing interactive visualization...")
        visualizer.visualize_features(img, processed_data)
        
        print(f"\nVisualizations saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 