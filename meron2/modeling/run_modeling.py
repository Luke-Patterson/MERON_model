import argparse
from pathlib import Path
from landmark_model import LandmarkModel

def main():
    # Hardcoded arguments
    features_path = 'data/processed/landmarks/landmark_features.csv'
    target_path = 'data/processed/malnutrition_flags.csv'
    target_var = 'malnutrition'
    output_dir = 'meron2/modeling/experiments'
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run modeling pipeline
    model = LandmarkModel(
        features_path=features_path,
        target_path=target_path,
        output_dir=output_dir
    )
    
    try:
        model.run_pipeline(target_var)
        print(f"Modeling pipeline completed successfully. Results saved to {output_dir}")
    except Exception as e:
        print(f"Error running modeling pipeline: {str(e)}")

if __name__ == "__main__":
    main() 