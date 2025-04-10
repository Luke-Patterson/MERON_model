"""
Model Comparison Script

This script runs all available models and compares their performance on the malnutrition classification task.
It standardizes the evaluation metrics and generates comparison visualizations.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any
import importlib.util
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meron2/modeling/experiments/model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path('meron2/modeling/experiments/model_comparison')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard metrics to collect from each model
STANDARD_METRICS = [
    'accuracy',
    'f1_score',
    'precision',
    'recall',
    'roc_auc',
    'confusion_matrix',
    'training_time',
    'inference_time',
    'roc_curve'  # Added ROC curve data
]

class ModelRunner:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.results = {}
        
    def run_model(self) -> Dict[str, Any]:
        """Run the model and collect standardized metrics"""
        logger.info(f"Running {self.model_name}...")
        
        # Import the model module
        spec = importlib.util.spec_from_file_location("model_module", self.model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["model_module"] = model_module
        spec.loader.exec_module(model_module)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the model
            if hasattr(model_module, 'main'):
                model_module.main()
            else:
                raise AttributeError(f"Model {self.model_name} does not have a main() function")
            
            # Record training time
            training_time = time.time() - start_time
            
            # Collect metrics (this will need to be adapted based on each model's output format)
            metrics = {
                'model_name': self.model_name,
                'training_time': training_time,
                'inference_time': None,  # Will be updated if available
                'accuracy': None,
                'f1_score': None,
                'precision': None,
                'recall': None,
                'roc_auc': None,
                'confusion_matrix': None
            }
            
            # Update metrics based on model output
            # This is a placeholder - actual implementation will depend on how each model returns results
            self.results = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running {self.model_name}: {str(e)}")
            return {
                'model_name': self.model_name,
                'error': str(e)
            }

def load_model_results(model_path: str) -> Dict[str, Any]:
    """Load results from a model's output file"""
    try:
        with open(model_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {model_path}: {str(e)}")
        return {}

def compare_models(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compare model performance and create a summary DataFrame"""
    # Create a DataFrame from the results
    df = pd.DataFrame(model_results)
    
    # Calculate additional metrics
    df['f1_precision_recall_avg'] = df[['f1_score', 'precision', 'recall']].mean(axis=1)
    
    # Sort by F1 score
    df = df.sort_values('f1_score', ascending=False)
    
    return df

def visualize_comparison(df: pd.DataFrame, output_dir: Path):
    """Create visualizations comparing model performance"""
    # Set style
    plt.style.use('seaborn')
    
    # 1. Bar plot of F1 scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='f1_score', data=df)
    plt.title('Model F1 Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_scores.png')
    plt.close()
    
    # 2. Radar plot of key metrics
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    plt.figure(figsize=(10, 10))
    for idx, row in df.iterrows():
        values = [row[metric] for metric in metrics]
        values = np.concatenate((values, [values[0]]))
        plt.polar(angles, values, label=row['model_name'])
    
    plt.xticks(angles[:-1], metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_plot.png')
    plt.close()
    
    # 3. Training time comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model_name', y='training_time', data=df)
    plt.title('Model Training Times')
    plt.xticks(rotation=45)
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_times.png')
    plt.close()
    
    # 4. ROC curves comparison
    plt.figure(figsize=(10, 8))
    for idx, row in df.iterrows():
        if 'roc_curve' in row and row['roc_curve'] is not None:
            fpr, tpr, _ = row['roc_curve']
            auc = row['roc_auc']
            plt.plot(fpr, tpr, label=f"{row['model_name']} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png')
    plt.close()

def collect_roc_data(model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect ROC curve data from each model's results"""
    for result in model_results:
        if 'roc_curve' not in result:
            # Try to find ROC curve data in the model's output files
            model_dir = Path('meron2/modeling/experiments') / result['model_name'].lower().replace(' ', '_')
            if model_dir.exists():
                # Look for ROC curve data in the model's output files
                for file in model_dir.glob('*_metrics.json'):
                    try:
                        with open(file, 'r') as f:
                            metrics = json.load(f)
                            if 'roc_curve' in metrics:
                                result['roc_curve'] = metrics['roc_curve']
                                result['roc_auc'] = metrics.get('roc_auc', None)
                                break
                    except Exception as e:
                        logger.warning(f"Error loading ROC data from {file}: {str(e)}")
    
    return model_results

def save_comparison_results(df: pd.DataFrame, output_dir: Path):
    """Save comparison results to files"""
    # Save DataFrame to CSV
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Save summary statistics
    summary = {
        'best_model': df.iloc[0]['model_name'],
        'best_f1_score': df.iloc[0]['f1_score'],
        'average_f1_score': df['f1_score'].mean(),
        'model_count': len(df),
        'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    """Main function to run all models and compare their performance"""
    logger.info("Starting model comparison")
    
    # Define models to run
    models = [
        {
            'path': 'meron2/modeling/aanjankumar_resnet50_malnutrition.py',
            'name': 'ResNet50'
        },
        {
            'path': 'meron2/modeling/advanced_ensemble_model.py',
            'name': 'Advanced Ensemble'
        },
        {
            'path': 'meron2/modeling/landmark_model.py',
            'name': 'Landmark Model'
        },
        {
            'path': 'meron2/modeling/multimodal_fusion_model.py',
            'name': 'Multimodal Fusion'
        },
        {
            'path': 'meron2/modeling/neural_network_model_20250327_improved.py',
            'name': 'Improved Neural Network'
        },
        {
            'path': 'meron2/modeling/vit_model.py',
            'name': 'Vision Transformer'
        },
        {
            'path': 'meron2/modeling/xgboost_model.py',
            'name': 'XGBoost'
        }
    ]
    
    # Create timestamp for this comparison run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # Run each model and collect results
    all_results = []
    for model_info in models:
        runner = ModelRunner(model_info['path'], model_info['name'])
        results = runner.run_model()
        all_results.append(results)
    
    # Collect ROC curve data
    all_results = collect_roc_data(all_results)
    
    # Compare models
    comparison_df = compare_models(all_results)
    
    # Create visualizations
    visualize_comparison(comparison_df, run_dir)
    
    # Save results
    save_comparison_results(comparison_df, run_dir)
    
    logger.info(f"Model comparison completed. Results saved to {run_dir}")
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(f"Best Model: {comparison_df.iloc[0]['model_name']}")
    print(f"Best F1 Score: {comparison_df.iloc[0]['f1_score']:.4f}")
    print(f"Average F1 Score: {comparison_df['f1_score'].mean():.4f}")
    print("\nDetailed Results:")
    print(comparison_df[['model_name', 'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'training_time']])

if __name__ == "__main__":
    main() 