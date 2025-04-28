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
from base_model import BaseModel

class LandmarkModel(BaseModel):
    """Modeling pipeline for facial landmark-based malnutrition prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the modeling pipeline.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'features_path': 'data/processed/landmarks/landmark_features.csv',
            'target_path': 'data/processed/malnutrition_flags.csv',
            'model_params': {
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            }
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('landmark', default_config)
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_distribution)
        """
        try:
            logging.info("Loading landmark features...")
            # Load landmark features
            features_path = Path(self.config['features_path'])
            landmark_features_df = pd.read_csv(features_path)
            logging.info(f"Landmark features shape: {landmark_features_df.shape}")
            
            logging.info("Loading geometric features...")
            # Load geometric features
            geometric_features_path = features_path.parent / 'geometric_features.csv'
            geometric_features_df = pd.read_csv(geometric_features_path)
            logging.info(f"Geometric features shape: {geometric_features_df.shape}")
            
            # Get feature column names
            landmark_feature_names = [col for col in landmark_features_df.columns if col != 'photo_id']
            geometric_feature_names = [col for col in geometric_features_df.columns if col != 'photo_id']
            self.feature_names = landmark_feature_names + geometric_feature_names
            logging.info(f"Number of landmark features: {len(landmark_feature_names)}")
            logging.info(f"Number of geometric features: {len(geometric_feature_names)}")
            
            logging.info("Combining features...")
            # Combine features
            features_df = pd.merge(landmark_features_df, geometric_features_df, on='photo_id', how='inner')
            logging.info(f"Combined features shape: {features_df.shape}")
            
            logging.info("Loading targets...")
            # Load targets
            targets_df = pd.read_csv(self.config['target_path'])
            logging.info(f"Targets shape: {targets_df.shape}")
            
            # Ensure photo_id is string type and remove .jpg extension from targets
            features_df['photo_id'] = features_df['photo_id'].astype(str)
            targets_df['photo_id'] = targets_df['photo_id'].astype(str).str.replace('.jpg', '')
            
            logging.info("Merging features and targets...")
            # Merge features and targets on photo_id
            merged_df = pd.merge(features_df, targets_df, on='photo_id', how='inner')
            logging.info(f"Merged shape: {merged_df.shape}")
            
            # Set features
            X = merged_df[self.feature_names].values
            logging.info(f"Final X shape: {X.shape}")
            
            # Create binary target variable
            y = merged_df.apply(
                lambda row: 1 if row['sam'] == 1 or row['mam'] == 1 else 0,
                axis=1
            ).values
            
            # Get class distribution
            class_distribution = pd.Series(y).value_counts().sort_index()
            logging.info(f"Class distribution:\n{class_distribution}")
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            logging.info(f"Training set shape: {X_train.shape}")
            logging.info(f"Test set shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, class_distribution
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def train(self):
        """
        Train the model with multiple algorithms and select the best one.
        
        Returns:
            tuple: (X_test, y_test) for evaluation
        """
        # Load data
        X_train, X_test, y_train, y_test, class_distribution = self.load_data()
        
        # Calculate class weights
        total_samples = class_distribution.sum()
        class_weights = {
            i: total_samples / (len(class_distribution) * count)
            for i, count in enumerate(class_distribution)
        }
        logging.info(f"Class weights: {class_weights}")
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        
        # Train multiple models
        for name, params in self.config['model_params'].items():
            logging.info(f"Training {name} model...")
            
            if name == 'random_forest':
                model = RandomForestClassifier(random_state=self.config['random_state'], class_weight='balanced')
            elif name == 'gradient_boosting':
                model = GradientBoostingClassifier(random_state=self.config['random_state'])
            elif name == 'logistic_regression':
                model = LogisticRegression(random_state=self.config['random_state'], class_weight='balanced', solver='liblinear')
            else:
                continue
                
            # Use F1 score for scoring
            grid_search = GridSearchCV(
                model,
                params,
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
                self.best_params = grid_search.best_params_
        
        self.model = best_model
        logging.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        
        return X_test, y_test
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        # Scale features if scaler is available
        if hasattr(self, 'scaler'):
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Make probability predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        # Scale features if scaler is available
        if hasattr(self, 'scaler'):
            X = self.scaler.transform(X)
            
        return self.model.predict_proba(X)
        
    def save_model(self):
        """Save the model and scaler to disk."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Save model
        model_path = self.model_dir / 'model.joblib'
        joblib.dump(self.model, model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
        
        # Save best parameters
        params_path = self.model_dir / 'best_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        logging.info(f"Best parameters saved to {params_path}")
        
        # Save feature names
        feature_names_path = self.model_dir / 'feature_names.json'
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=4)
        logging.info(f"Feature names saved to {feature_names_path}")
        
    def load_model(self, model_path):
        """
        Load the model and scaler from disk.
        
        Args:
            model_path (str): Path to the model directory
        """
        model_path = Path(model_path)
        
        # Load model
        self.model = joblib.load(model_path / 'model.joblib')
        logging.info(f"Model loaded from {model_path / 'model.joblib'}")
        
        # Load scaler
        self.scaler = joblib.load(model_path / 'scaler.joblib')
        logging.info(f"Scaler loaded from {model_path / 'scaler.joblib'}")
        
        # Load best parameters
        with open(model_path / 'best_params.json', 'r') as f:
            self.best_params = json.load(f)
        logging.info(f"Best parameters loaded from {model_path / 'best_params.json'}")
        
        # Load feature names
        with open(model_path / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        logging.info(f"Feature names loaded from {model_path / 'feature_names.json'}")
        
def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = LandmarkModel()
    
    # Train model
    X_test, y_test = model.train()
    
    # Save model
    model.save_model()
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    # Visualize results
    model.visualize_results(results)
    
if __name__ == "__main__":
    main() 