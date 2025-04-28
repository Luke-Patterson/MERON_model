import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from base_model import BaseModel
import json

class XGBoostModel(BaseModel):
    """XGBoost model for malnutrition classification."""
    
    def __init__(self, config=None):
        """
        Initialize the XGBoost model.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'feature_cols_start': 1,  # The ResNet50 features start from column 1
            'feature_cols_end': 2049,  # The ResNet50 features end at column 2048
            'data_path': os.path.join('data', 'processed', 'features_with_flags.csv'),
            'model_params': {
                'colsample_bytree': 0.8,
                'learning_rate': 0.01,
                'max_depth': 3,
                'n_estimators': 1000,
                'subsample': 1.0,
                'scale_pos_weight': 4.0,  # This will give more weight to the minority class (0.2/0.8 = 4)
                'min_child_weight': 1,
                'gamma': 0.1
            }
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('xgboost', default_config)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_distribution)
        """
        logging.info("Loading data...")
        df = pd.read_csv(self.config['data_path'])
        
        # Create binary target variable
        logging.info("Creating binary target variable...")
        df['malnutrition_class'] = df.apply(
            lambda row: 1 if row['sam'] == 1 or row['mam'] == 1 else 0,
            axis=1
        )
        
        # Get class distribution
        class_distribution = df['malnutrition_class'].value_counts().sort_index()
        logging.info(f"Class distribution:\n{class_distribution}")
        
        # Select features (ResNet50 features)
        X = df.iloc[:, self.config['feature_cols_start']:self.config['feature_cols_end']].values
        
        # Select target
        y = df['malnutrition_class'].values
        
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
        
    def train(self):
        """
        Train the XGBoost model with fixed hyperparameters.
        
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
        
        # Set up the XGBoost classifier with fixed parameters
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.config['random_state'],
            eval_metric='logloss',
            **self.config['model_params']
        )
        
        # Apply class weights to samples
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        # Fit the model
        logging.info("Training XGBoost model...")
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store the parameters used
        self.best_params = self.config['model_params']
        logging.info(f"Model parameters: {self.best_params}")
        
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
        
def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = XGBoostModel()
    
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