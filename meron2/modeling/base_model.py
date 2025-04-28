import os
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

class BaseModel:
    """Base class for all malnutrition prediction models."""
    
    def __init__(self, model_name, config=None):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name of the model (e.g., 'resnet50', 'xgboost')
            config (dict): Model configuration parameters
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        
        # Setup directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Setup model and results directories."""
        # Create base directories
        self.models_dir = Path('meron2/modeling/models')
        self.results_dir = Path('meron2/modeling/results')
        
        # Create model-specific directories
        self.model_dir = self.models_dir / self.model_name
        self.model_results_dir = self.results_dir / self.model_name
        
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.results_dir, self.model_dir, self.model_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.model_results_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Save configuration
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def load_data(self):
        """
        Load and preprocess the data.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement load_data()")
        
    def train(self):
        """
        Train the model.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement train()")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Make predictions
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_per_class': f1_score(y_test, y_pred, average=None).tolist(),
            'precision_per_class': precision_score(y_test, y_pred, average=None).tolist(),
            'recall_per_class': recall_score(y_test, y_pred, average=None).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add ROC AUC and curve data if probabilities are available
        if y_prob is not None:
            try:
                # Calculate ROC curve data
                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
                # Calculate AUC
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
            except:
                metrics['roc_auc'] = None
                metrics['roc_curve'] = None
                
        return metrics
        
    def predict(self, X):
        """
        Make predictions.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement predict()")
        
    def predict_proba(self, X):
        """
        Make probability predictions.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement predict_proba()")
        
    def save_model(self):
        """
        Save the model to disk.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement save_model()")
        
    def load_model(self, model_path):
        """
        Load the model from disk.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement load_model()")
        
    def visualize_results(self, results):
        """
        Visualize model results.
        
        Args:
            results (dict): Dictionary containing evaluation metrics
        """
        # Create figure for confusion matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.model_results_dir / 'confusion_matrix.png')
        plt.close()
        
        # Create figure for ROC curve if available
        if 'roc_curve' in results and results['roc_curve'] is not None:
            plt.figure(figsize=(10, 8))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(
                results['roc_curve']['fpr'],
                results['roc_curve']['tpr'],
                label=f'ROC (AUC = {results["roc_auc"]:.3f})'
            )
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(self.model_results_dir / 'roc_curve.png')
            plt.close()
            
        # Save metrics to JSON
        with open(self.model_results_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4) 