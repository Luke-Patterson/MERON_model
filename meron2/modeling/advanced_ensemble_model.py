import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from base_model import BaseModel
import joblib
from datetime import datetime
import optuna
from collections import Counter
import torch.nn.functional as F

class MalnutritionDataset(Dataset):
    """PyTorch dataset for malnutrition classification."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MalnutritionNN(nn.Module):
    """Neural network architecture for malnutrition classification."""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(MalnutritionNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer for binary classification
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.layers.append(nn.Sigmoid())  # Sigmoid for binary classification

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg

class AdvancedEnsembleModel(BaseModel):
    """Advanced ensemble model for malnutrition classification."""
    
    def __init__(self, config=None):
        """
        Initialize the advanced ensemble model.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'data_path': os.path.join('data', 'processed', 'features_with_flags.csv'),
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'l1_lambda': 1e-5,
            'feature_selection_k': 1000,
            'pca_components': 300,
            'n_splits': 5,
            'patience': 10,
            'class_weights': [1.0, 3.0],
            'use_smote': True,
            'smote_ratio': 0.8,
            'n_trials': 20,
            'n_models': 5
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('advanced_ensemble', default_config)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=self.config['feature_selection_k'])
        self.pca = PCA(n_components=self.config['pca_components'], random_state=self.config['random_state'])
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.best_params = None
        
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_distribution, X_train_processed, X_test_processed)
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
        X = df.iloc[:, 1:2049].values  # ResNet50 features
        
        # Select target
        y = df['malnutrition_class'].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Store original data
        X_train_original = X_train.copy()
        X_test_original = X_test.copy()
        
        # Preprocess features in the correct order
        # 1. Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # 2. Select features
        X_train = self.feature_selector.fit_transform(X_train, y_train)
        X_test = self.feature_selector.transform(X_test)
        
        # 3. Apply PCA
        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        return X_train_original, X_test_original, y_train, y_test, class_distribution, X_train, X_test
        
    def train(self):
        """
        Train the ensemble model.
        
        Returns:
            tuple: (X_test_original, y_test) for evaluation
        """
        # Load data
        X_train_original, X_test_original, y_train, y_test, class_distribution, X_train, X_test = self.load_data()
        
        # Apply SMOTE if enabled
        if self.config['use_smote']:
            smote = SMOTE(sampling_strategy=self.config['smote_ratio'], random_state=self.config['random_state'])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE - Training set shape: {X_train.shape}")
        
        # Create cross-validation folds
        kf = StratifiedKFold(n_splits=self.config['n_splits'], shuffle=True, random_state=self.config['random_state'])
        
        # Initialize lists to store models and their parameters
        self.models = []
        fold_params = []
        fold_scores = []
        
        # Train models for each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            logging.info(f"\nTraining model for fold {fold+1}/{self.config['n_splits']}")
            
            # Split data for this fold
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]
            
            # Hyperparameter optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X_fold_train, y_fold_train, X_fold_val, y_fold_val),
                n_trials=self.config['n_trials']
            )
            
            # Get best hyperparameters
            best_params = study.best_params
            logging.info(f"Best hyperparameters for fold {fold+1}: {best_params}")
            fold_params.append(best_params)
            
            # Train model with best hyperparameters
            model = self._train_model(X_fold_train, y_fold_train, best_params)
            self.models.append(model)
            
            # Evaluate on validation set
            val_preds = self._predict_model(model, X_fold_val)
            val_f1 = f1_score(y_fold_val, val_preds)
            fold_scores.append(val_f1)
            logging.info(f"Validation F1 score for fold {fold+1}: {val_f1:.4f}")
        
        # Store the best parameters from the fold with highest validation score
        best_fold_idx = np.argmax(fold_scores)
        self.best_params = fold_params[best_fold_idx]
        logging.info(f"Best fold: {best_fold_idx + 1} with F1 score: {fold_scores[best_fold_idx]:.4f}")
        logging.info(f"Using parameters from best fold: {self.best_params}")
        
        # Set the best model as the main model for compatibility with base_model
        self.model = self.models[best_fold_idx]
        
        return X_test_original, y_test
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for Optuna hyperparameter optimization."""
        # Define hyperparameters to optimize
        hidden_size1 = trial.suggest_int('hidden_size1', 64, 512)
        hidden_size2 = trial.suggest_int('hidden_size2', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 1e-4, log=True)
        
        # Create model with suggested hyperparameters
        model = MalnutritionNN(
            input_size=X_train.shape[1],
            hidden_sizes=[hidden_size1, hidden_size2, 1],
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Create dataset and data loader
        train_dataset = MalnutritionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Define loss function with class weights
        pos_weight = torch.tensor([self.config['class_weights'][1] / self.config['class_weights'][0]]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Train for a few epochs
        model.train()
        for epoch in range(10):  # Limited epochs for hyperparameter search
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(features)
                # Reshape labels to match output shape
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                
                # Add L1 regularization
                l1_reg = model.l1_regularization()
                loss = loss + l1_lambda * l1_reg
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_preds = self._predict_model(model, X_val)
            val_f1 = f1_score(y_val, val_preds)
        
        return val_f1
        
    def _train_model(self, X_train, y_train, params):
        """Train a single model with given parameters."""
        # Create model
        model = MalnutritionNN(
            input_size=X_train.shape[1],
            hidden_sizes=[params['hidden_size1'], params['hidden_size2'], 1],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Create dataset and data loader
        train_dataset = MalnutritionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Define loss function with class weights
        pos_weight = torch.tensor([self.config['class_weights'][1] / self.config['class_weights'][0]]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Define optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            model.train()
            train_loss = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(features)
                # Reshape labels to match output shape
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                
                # Add L1 regularization
                l1_reg = model.l1_regularization()
                loss = loss + params['l1_lambda'] * l1_reg
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                break
                
            patience_counter += 1
        
        return model
        
    def _predict_model(self, model, X):
        """Make predictions using a single model."""
        model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            outputs = model(X)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            # Use a higher threshold for positive class due to imbalance
            predictions = (probs > 0.6).float()
            return predictions.cpu().numpy(), probs.cpu().numpy()
        
    def predict(self, X):
        """
        Make predictions using the ensemble.
        
        Args:
            X: Features to predict on (raw data)
            
        Returns:
            array: Predicted labels
        """
        if not self.models:
            raise ValueError("Model must be trained before prediction")
            
        # Preprocess features in the correct order
        X = self.scaler.transform(X)  # Scale first
        X = self.feature_selector.transform(X)  # Then select features
        X = self.pca.transform(X)  # Finally apply PCA
        
        # Get predictions from each model
        predictions = []
        probabilities = []
        for model in self.models:
            pred, prob = self._predict_model(model, X)
            predictions.append(pred)
            probabilities.append(prob)
        
        # Ensemble predictions (majority voting)
        ensemble_pred = np.mean(predictions, axis=0) > 0.5
        return ensemble_pred.astype(int)
        
    def predict_proba(self, X):
        """
        Make probability predictions using the ensemble.
        
        Args:
            X: Features to predict on (raw data)
            
        Returns:
            array: Predicted probabilities
        """
        if not self.models:
            raise ValueError("Model must be trained before prediction")
            
        # Preprocess features in the correct order
        X = self.scaler.transform(X)  # Scale first
        X = self.feature_selector.transform(X)  # Then select features
        X = self.pca.transform(X)  # Finally apply PCA
        
        # Get probabilities from each model
        probabilities = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
                probabilities.append(probs)
        
        # Ensemble probabilities (average)
        ensemble_prob = np.mean(probabilities, axis=0)
        return ensemble_prob
        
    def save_model(self):
        """Save the model and preprocessing components to disk."""
        if not self.models:
            raise ValueError("Model must be trained before saving")
            
        # Save models
        for i, model in enumerate(self.models):
            model_path = self.model_dir / f'model_{i+1}.pt'
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model {i+1} saved to {model_path}")
        
        # Save preprocessing components
        joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
        joblib.dump(self.feature_selector, self.model_dir / 'feature_selector.joblib')
        joblib.dump(self.pca, self.model_dir / 'pca.joblib')
        
        # Save best parameters
        with open(self.model_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        logging.info("Model components saved")
        
    def load_model(self, model_path):
        """
        Load the model and preprocessing components from disk.
        
        Args:
            model_path (str): Path to the model directory
        """
        model_path = Path(model_path)
        
        # Load preprocessing components
        self.scaler = joblib.load(model_path / 'scaler.joblib')
        self.feature_selector = joblib.load(model_path / 'feature_selector.joblib')
        self.pca = joblib.load(model_path / 'pca.joblib')
        
        # Load best parameters
        with open(model_path / 'best_params.json', 'r') as f:
            self.best_params = json.load(f)
        
        # Load models
        self.models = []
        for i in range(self.config['n_models']):
            model = MalnutritionNN(
                input_size=self.config['pca_components'],
                hidden_sizes=[
                    self.best_params['hidden_size1'],
                    self.best_params['hidden_size2'],
                    1
                ],
                dropout_rate=self.best_params['dropout_rate']
            ).to(self.device)
            model.load_state_dict(torch.load(model_path / f'model_{i+1}.pt'))
            self.models.append(model)
        
        logging.info(f"Model and components loaded from {model_path}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        logging.info("Evaluating model performance...")
        results = super().evaluate(X_test, y_test)
        
        # Log key metrics
        logging.info(f"Accuracy: {results['accuracy']:.4f}")
        logging.info(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        logging.info(f"Precision (Macro): {results['precision_macro']:.4f}")
        logging.info(f"Recall (Macro): {results['recall_macro']:.4f}")
        
        if results['roc_auc'] is not None:
            logging.info(f"ROC AUC: {results['roc_auc']:.4f}")
            logging.info("ROC curve data saved in metrics.json")
        else:
            logging.warning("ROC AUC could not be calculated")
            
        return results
        
    def visualize_results(self, results):
        """
        Visualize model results.
        
        Args:
            results (dict): Dictionary containing evaluation metrics
        """
        logging.info("Visualizing model results...")
        super().visualize_results(results)
        logging.info(f"Results saved in {self.model_results_dir}")

def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = AdvancedEnsembleModel()
    
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