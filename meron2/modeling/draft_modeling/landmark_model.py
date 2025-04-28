import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from datetime import datetime

class LandmarkModel:
    """Modeling pipeline for facial landmark-based malnutrition prediction."""
    
    def __init__(self, features_path: str, target_path: str, output_dir: str = "model_output"):
        """
        Initialize the modeling pipeline.
        
        Args:
            features_path: Path to CSV file containing facial landmark features
            target_path: Path to CSV file containing target variables
            output_dir: Directory to save model outputs
        """
        self.features_path = Path(features_path)
        self.target_path = Path(target_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(
            filename=str(self.output_dir / 'modeling.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize data attributes
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_params = None
        
    def load_data(self) -> None:
        """Load and preprocess the feature and target data."""
        try:
            print("Loading landmark features...")
            # Load landmark features
            features_path = Path(self.features_path)
            landmark_features_df = pd.read_csv(features_path)
            print(f"Landmark features shape: {landmark_features_df.shape}")
            
            print("Loading geometric features...")
            # Load geometric features
            geometric_features_path = features_path.parent / 'geometric_features.csv'
            geometric_features_df = pd.read_csv(geometric_features_path)
            print(f"Geometric features shape: {geometric_features_df.shape}")
            
            # Get feature column names
            landmark_feature_names = [col for col in landmark_features_df.columns if col != 'photo_id']
            geometric_feature_names = [col for col in geometric_features_df.columns if col != 'photo_id']
            print(f"Number of landmark features: {len(landmark_feature_names)}")
            print(f"Number of geometric features: {len(geometric_feature_names)}")
            
            print("Combining features...")
            # Combine features
            features_df = pd.merge(landmark_features_df, geometric_features_df, on='photo_id', how='inner')
            self.feature_names = landmark_feature_names + geometric_feature_names
            print(f"Combined features shape: {features_df.shape}")
            
            print("Loading targets...")
            # Load targets
            targets_df = pd.read_csv(self.target_path)
            print(f"Targets shape: {targets_df.shape}")
            
            # Ensure photo_id is string type and remove .jpg extension from targets
            features_df['photo_id'] = features_df['photo_id'].astype(str)
            targets_df['photo_id'] = targets_df['photo_id'].astype(str).str.replace('.jpg', '')
            
            print("Merging features and targets...")
            # Merge features and targets on photo_id
            merged_df = pd.merge(features_df, targets_df, on='photo_id', how='inner')
            print(f"Merged shape: {merged_df.shape}")
            
            # Set features
            self.X = merged_df[self.feature_names].values
            print(f"Final X shape: {self.X.shape}")
            
            # Define target variables mapping
            self.target_vars = {
                'sam': 'sam',
                'mam': 'mam',
                'malnutrition': 'malnutrition',
                'whz': 'whz',
                'muac_cm': 'muac_cm'
            }
            
            # Log data shapes and info
            logging.info(f"Landmark features shape: {landmark_features_df.shape}")
            logging.info(f"Geometric features shape: {geometric_features_df.shape}")
            logging.info(f"Combined features shape: {features_df.shape}")
            logging.info(f"Targets shape: {targets_df.shape}")
            logging.info(f"Final merged shape: {merged_df.shape}")
            logging.info(f"Number of landmark features: {len(landmark_feature_names)}")
            logging.info(f"Number of geometric features: {len(geometric_feature_names)}")
            logging.info(f"Total number of features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, target_var: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Preprocess data for modeling.
        
        Args:
            target_var: Target variable to predict
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logging.info(f"Training set size: {len(X_train_scaled)}")
            logging.info(f"Test set size: {len(X_test_scaled)}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _get_model_configs(self):
        """Get model configurations with class weights and F1 optimization."""
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'),
                'param_grid': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            }
        }
    
    def _train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one."""
        logging.info(f"Training set size: {len(X_train)}")
        logging.info(f"Test set size: {len(X_test)}")
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        self.models = {}  # Initialize models dictionary
        
        for name, config in self._get_model_configs().items():
            logging.info(f"Training {name} model...")
            
            # Use F1 score for scoring
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
            
            cv_score = grid_search.best_score_
            test_score = grid_search.score(X_test, y_test)
            
            # Store the trained model
            self.models[name] = {
                'model': grid_search.best_estimator_,
                'cv_score': cv_score,
                'test_score': test_score,
                'params': grid_search.best_params_
            }
            
            logging.info(f"{name} best parameters: {grid_search.best_params_}")
            logging.info(f"{name} CV score: {cv_score:.4f}")
            logging.info(f"{name} test score: {test_score:.4f}")
            
            if cv_score > best_score:
                best_score = cv_score
                best_model = grid_search.best_estimator_
                best_model_name = name
                self.cv_score = cv_score  # Store best CV score
                self.best_params = grid_search.best_params_  # Store best parameters
        
        logging.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        return best_model, best_model_name
    
    def _save_model(self, model, model_name):
        """Save the trained model with timestamped subfolder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.output_dir, f"{model_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        
        # Save evaluation metrics
        metrics_path = os.path.join(experiment_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.evaluation_metrics, f)
        
        logging.info(f"Saved model {model_name} to {experiment_dir}")
    
    def train_models(self, target_var: str, cv: int = 5) -> None:
        """
        Train and evaluate multiple models.
        
        Args:
            target_var: Target variable to predict
            cv: Number of cross-validation folds
        """
        try:
            # Get preprocessed data
            X_train, X_test, y_train, y_test = self.preprocess_data(target_var)
            
            # Train models
            self.best_model, self.best_model_name = self._train_models(X_train, y_train, X_test, y_test)
            
            # Evaluate model
            self.evaluation_metrics = self.evaluate_model(self.best_model_name, X_test, y_test)
            
            # Save model
            self._save_model(self.best_model, self.best_model_name)
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a trained model."""
        try:
            model = self.models[model_name]['model']
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            logging.info(f"Evaluation metrics for {model_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_name: str) -> None:
        """
        Save a trained model and its metadata.
        
        Args:
            model_name: Name of the model to save
        """
        try:
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            joblib.dump(
                self.models[model_name]['model'],
                model_dir / 'model.joblib'
            )
            
            # Save scaler
            joblib.dump(
                self.scaler,
                model_dir / 'scaler.joblib'
            )
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'best_params': self.models[model_name]['params'],
                'cv_score': self.models[model_name]['cv_score'],
                'test_score': self.models[model_name]['test_score'],
                'feature_names': self.feature_names
            }
            joblib.dump(
                metadata,
                model_dir / 'metadata.joblib'
            )
            
            logging.info(f"Saved model {model_name} to {model_dir}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def run_pipeline(self, target_var: str) -> None:
        """
        Run the complete modeling pipeline.
        
        Args:
            target_var: Target variable to predict
        """
        try:
            print(f"\nStarting pipeline for target variable: {target_var}")
            # Load data
            self.load_data()
            
            # Get target variable
            if target_var not in self.target_vars:
                raise ValueError(f"Invalid target variable: {target_var}")
            
            print("Loading targets...")
            # Load targets and merge with features
            targets_df = pd.read_csv(self.target_path)
            print(f"Targets shape: {targets_df.shape}")
            
            # Ensure photo_id is string type and remove .jpg extension
            targets_df['photo_id'] = targets_df['photo_id'].astype(str).str.replace('.jpg', '')
            
            # Get target values from the target variable mapping for matched samples only
            target_col = self.target_vars[target_var]
            
            # Create a features dataframe with photo_ids
            features_path = Path(self.features_path)
            landmark_features_df = pd.read_csv(features_path)
            geometric_features_path = features_path.parent / 'geometric_features.csv'
            geometric_features_df = pd.read_csv(geometric_features_path)
            features_df = pd.merge(landmark_features_df[['photo_id']], geometric_features_df[['photo_id']], on='photo_id', how='inner')
            features_df['photo_id'] = features_df['photo_id'].astype(str)
            
            # Merge with targets to get matched samples only
            merged_df = pd.merge(features_df[['photo_id']], targets_df, on='photo_id', how='inner')
            self.y = merged_df[target_col].values
            print(f"Target values shape: {self.y.shape}")
            
            print("Training models...")
            # Train models
            self.train_models(target_var)
            
            print("Evaluating model...")
            # Evaluate and save best model
            X_train, X_test, y_train, y_test = self.preprocess_data(target_var)
            metrics = self.evaluate_model(self.best_model_name, X_test, y_test)
            
            print("Saving model...")
            self._save_model(self.best_model, self.best_model_name)
            
            # Save summary
            summary = {
                'target_variable': target_var,
                'best_model': self.best_model_name,
                'best_params': self.best_params,
                'cv_score': self.cv_score,
                'test_metrics': metrics
            }
            joblib.dump(
                summary,
                self.output_dir / 'pipeline_summary.joblib'
            )
            
            print("Pipeline completed successfully")
            logging.info("Pipeline completed successfully")
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            logging.error(f"Error in pipeline: {str(e)}")
            raise 