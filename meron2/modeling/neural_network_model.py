import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import json
import logging
import joblib
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pathlib import Path
from base_model import BaseModel

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
    """Neural network model for malnutrition classification."""
    
    def __init__(self, input_size, num_classes=2):
        super(MalnutritionNN, self).__init__()
        
        # Moderate architecture with L1 regularization
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights to be small
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply L1 regularization by calculating L1 norm of weights
        l1_reg = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg += torch.sum(torch.abs(param))
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        # Return both the output and the L1 regularization term
        return x, l1_reg

class NeuralNetworkModel(BaseModel):
    """Neural network model for malnutrition classification."""
    
    def __init__(self, config=None):
        """
        Initialize the neural network model.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'validation_split': 0.2,
            'feature_cols_start': 1,
            'feature_cols_end': 2049,
            'data_path': os.path.join('data', 'processed', 'features_with_flags.csv'),
            'batch_size': 32,
            'num_epochs': 200,
            'learning_rate': 0.001,
            'early_stopping_patience': 25,
            'pca_components': 300,
            'balance_alpha': 0.9,
            'l1_weight': 1e-5,
            'class_weights': [1.0, 3.0]
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('neural_network', default_config)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=500)
        self.pca = PCA(n_components=self.config['pca_components'], random_state=self.config['random_state'])
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self):
        """
        Load and preprocess the data with dimensionality reduction.
        
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
        X = df.iloc[:, self.config['feature_cols_start']:self.config['feature_cols_end']].values
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
        
    def create_balanced_sampler(self, labels, alpha=0.9):
        """
        Create a weighted sampler that balances between natural distribution and complete balance.
        
        Args:
            labels: Training labels
            alpha: Controls how much we balance (0 = natural distribution, 1 = completely balanced)
            
        Returns:
            WeightedRandomSampler: Balanced sampler for training
        """
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Natural distribution
        natural_dist = class_counts / total_samples
        
        # Equal distribution
        equal_dist = np.ones_like(class_counts) / len(class_counts)
        
        # Blend distributions with alpha
        target_dist = (1 - alpha) * natural_dist + alpha * equal_dist
        
        # Calculate weights for each sample
        weights = np.zeros_like(labels, dtype=np.float32)
        for t in range(len(class_counts)):
            weights[labels == t] = target_dist[t] / class_counts[t]
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
    def train(self):
        """
        Train the neural network model.
        
        Returns:
            tuple: (X_test_original, y_test) for evaluation
        """
        # Load data
        X_train_original, X_test_original, y_train, y_test, class_distribution, X_train, X_test = self.load_data()
        
        # Split training data to create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['validation_split'],
            random_state=self.config['random_state'],
            stratify=y_train
        )
        
        # Create datasets
        train_dataset = MalnutritionDataset(X_train, y_train)
        val_dataset = MalnutritionDataset(X_val, y_val)
        test_dataset = MalnutritionDataset(X_test, y_test)
        
        # Create balanced sampler
        train_sampler = self.create_balanced_sampler(y_train, alpha=self.config['balance_alpha'])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]
        self.model = MalnutritionNN(input_size, num_classes=2).to(self.device)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create loss function with class weights
        class_weights = torch.FloatTensor(self.config['class_weights']).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-6
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Train model
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs, l1_reg = self.model(features)
                loss = criterion(outputs, labels) + self.config['l1_weight'] * l1_reg
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs, l1_reg = self.model(features)
                    loss = criterion(outputs, labels) + self.config['l1_weight'] * l1_reg
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            
            # Update learning rate scheduler
            scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            logging.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                        f"Train Loss: {train_loss:.4f} Train F1: {train_f1:.4f} "
                        f"Val Loss: {val_loss:.4f} Val F1: {val_f1:.4f}")
        
        # Load best model state
        self.model.load_state_dict(best_model_state)
        
        return X_test_original, y_test
        
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
            
        # Preprocess features in the correct order
        X = self.scaler.transform(X)  # Scale first
        X = self.feature_selector.transform(X)  # Then select features
        X = self.pca.transform(X)  # Finally apply PCA
        
        # Convert to tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
        
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
            
        # Preprocess features in the correct order
        X = self.scaler.transform(X)  # Scale first
        X = self.feature_selector.transform(X)  # Then select features
        X = self.pca.transform(X)  # Finally apply PCA
        
        # Convert to tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X)
            probs = torch.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
        
    def save_model(self):
        """Save the model and preprocessing components to disk."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Save model
        model_path = self.model_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save preprocessing components
        scaler_path = self.model_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
        
        feature_selector_path = self.model_dir / 'feature_selector.joblib'
        joblib.dump(self.feature_selector, feature_selector_path)
        logging.info(f"Feature selector saved to {feature_selector_path}")
        
        pca_path = self.model_dir / 'pca.joblib'
        joblib.dump(self.pca, pca_path)
        logging.info(f"PCA saved to {pca_path}")
        
        # Save configuration
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
        
    def load_model(self, model_path):
        """
        Load the model and preprocessing components from disk.
        
        Args:
            model_path (str): Path to the model directory
        """
        model_path = Path(model_path)
        
        # Load configuration
        with open(model_path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load preprocessing components
        self.scaler = joblib.load(model_path / 'scaler.joblib')
        self.feature_selector = joblib.load(model_path / 'feature_selector.joblib')
        self.pca = joblib.load(model_path / 'pca.joblib')
        
        # Create and load model
        input_size = self.config['pca_components']
        self.model = MalnutritionNN(input_size, num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path / 'model.pt'))
        
        logging.info(f"Model and components loaded from {model_path}")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        This method calculates various performance metrics including:
        - Accuracy
        - F1 score (macro and per-class)
        - Precision (macro and per-class)
        - Recall (macro and per-class)
        - ROC AUC and ROC curve
        - Confusion matrix
        
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
        
        This method creates and saves:
        - Confusion matrix plot
        - ROC curve plot (if available)
        - Metrics in JSON format
        
        Args:
            results (dict): Dictionary containing evaluation metrics
        """
        logging.info("Visualizing model results...")
        super().visualize_results(results)
        logging.info(f"Results saved in {self.model_results_dir}")
        
def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = NeuralNetworkModel()
    
    # Train model
    X_test_original, y_test = model.train()
    
    # Save model
    model.save_model()
    
    # Evaluate model
    results = model.evaluate(X_test_original, y_test)
    
    # Visualize results
    model.visualize_results(results)
    
if __name__ == "__main__":
    main() 