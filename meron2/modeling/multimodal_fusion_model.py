import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from base_model import BaseModel
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class MultimodalDataset(Dataset):
    """PyTorch dataset for multimodal malnutrition classification."""
    
    def __init__(self, resnet_features: np.ndarray, landmark_features: np.ndarray, labels: np.ndarray):
        self.resnet_features = torch.FloatTensor(resnet_features)
        self.landmark_features = torch.FloatTensor(landmark_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'resnet': self.resnet_features[idx],
            'landmarks': self.landmark_features[idx],
            'label': self.labels[idx]
        }

class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

class MultimodalFusionNN(nn.Module):
    """Neural network architecture for multimodal fusion."""
    
    def __init__(self, resnet_dim: int, landmark_dim: int, num_classes: int = 2):
        super().__init__()
        
        # ResNet feature processing
        self.resnet_attention = AttentionLayer(resnet_dim)
        self.resnet_fc = nn.Sequential(
            nn.Linear(resnet_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Landmark feature processing
        self.landmark_attention = AttentionLayer(landmark_dim)
        self.landmark_fc = nn.Sequential(
            nn.Linear(landmark_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, resnet_features, landmark_features):
        # Add noise during training
        if self.training:
            resnet_features = resnet_features + torch.randn_like(resnet_features) * 0.1
            landmark_features = landmark_features + torch.randn_like(landmark_features) * 0.1
        
        # Process ResNet features
        resnet_attended = self.resnet_attention(resnet_features)
        resnet_processed = self.resnet_fc(resnet_attended)
        
        # Process landmark features
        landmark_attended = self.landmark_attention(landmark_features)
        landmark_processed = self.landmark_fc(landmark_attended)
        
        # Concatenate and fuse
        combined = torch.cat([resnet_processed, landmark_processed], dim=1)
        output = self.fusion(combined)
        
        return output

class MultimodalFusionModel(BaseModel):
    """Multimodal fusion model for malnutrition classification."""
    
    def __init__(self, config=None):
        """
        Initialize the multimodal fusion model.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'resnet_features_path': 'data/processed/resnet50_features.csv',
            'landmark_features_path': 'data/processed/landmarks/landmark_features.csv',
            'labels_path': 'data/processed/malnutrition_flags.csv',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 50,
            'patience': 10,
            'weight_decay': 1e-4,
            'model_params': {
                'resnet_dim': 2048,
                'landmark_dim': 141,  # Updated to match actual feature dimension
                'num_classes': 2
            }
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('multimodal_fusion', default_config)
        
        # Initialize preprocessing components
        self.resnet_scaler = StandardScaler()
        self.landmark_scaler = StandardScaler()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_distribution)
        """
        logging.info("Loading data...")
        
        # Load features and labels
        resnet_features = pd.read_csv(self.config['resnet_features_path'])
        landmark_features = pd.read_csv(self.config['landmark_features_path'])
        labels = pd.read_csv(self.config['labels_path'])
        
        # Clean photo IDs
        resnet_features['photo_id'] = resnet_features['photo_id'].str.replace('.jpg', '')
        landmark_features['photo_id'] = landmark_features['photo_id'].str.replace('.jpg', '')
        labels['photo_id'] = labels['photo_id'].str.replace('.jpg', '')
        
        # Align data
        common_ids = set(resnet_features['photo_id']).intersection(
            set(landmark_features['photo_id']).intersection(set(labels['photo_id']))
        )
        
        resnet_features = resnet_features[resnet_features['photo_id'].isin(common_ids)]
        landmark_features = landmark_features[landmark_features['photo_id'].isin(common_ids)]
        labels = labels[labels['photo_id'].isin(common_ids)]
        
        # Prepare features and labels
        X_resnet = resnet_features.drop('photo_id', axis=1).values
        X_landmarks = landmark_features.drop('photo_id', axis=1).values
        y = labels['malnutrition'].values
        
        # Verify feature dimensions
        if X_resnet.shape[1] != self.config['model_params']['resnet_dim']:
            raise ValueError(f"ResNet features dimension mismatch. Expected {self.config['model_params']['resnet_dim']}, got {X_resnet.shape[1]}")
        if X_landmarks.shape[1] != self.config['model_params']['landmark_dim']:
            raise ValueError(f"Landmark features dimension mismatch. Expected {self.config['model_params']['landmark_dim']}, got {X_landmarks.shape[1]}")
        
        # Get class distribution
        class_distribution = pd.Series(y).value_counts().sort_index()
        logging.info(f"Class distribution:\n{class_distribution}")
        
        # Split data
        X_resnet_train, X_resnet_test, X_landmarks_train, X_landmarks_test, y_train, y_test = train_test_split(
            X_resnet, X_landmarks, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features
        X_resnet_train = self.resnet_scaler.fit_transform(X_resnet_train)
        X_resnet_test = self.resnet_scaler.transform(X_resnet_test)
        
        X_landmarks_train = self.landmark_scaler.fit_transform(X_landmarks_train)
        X_landmarks_test = self.landmark_scaler.transform(X_landmarks_test)
        
        logging.info(f"Training set shape: ResNet {X_resnet_train.shape}, Landmarks {X_landmarks_train.shape}")
        logging.info(f"Test set shape: ResNet {X_resnet_test.shape}, Landmarks {X_landmarks_test.shape}")
        
        return (X_resnet_train, X_landmarks_train), (X_resnet_test, X_landmarks_test), y_train, y_test, class_distribution
        
    def train(self):
        """
        Train the multimodal fusion model.
        
        Returns:
            tuple: (X_test, y_test) for evaluation
        """
        # Load data
        (X_resnet_train, X_landmarks_train), (X_resnet_test, X_landmarks_test), y_train, y_test, class_distribution = self.load_data()
        
        # Create datasets
        train_dataset = MultimodalDataset(X_resnet_train, X_landmarks_train, y_train)
        test_dataset = MultimodalDataset(X_resnet_test, X_landmarks_test, y_test)
        
        # Create weighted sampler for class imbalance
        class_weights = torch.FloatTensor(
            len(y_train) / (2 * np.bincount(y_train))
        )
        train_sampler = WeightedRandomSampler(
            weights=class_weights[y_train],
            num_samples=len(y_train),
            replacement=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']
        )
        
        # Initialize model
        self.model = MultimodalFusionNN(
            resnet_dim=self.config['model_params']['resnet_dim'],
            landmark_dim=self.config['model_params']['landmark_dim'],
            num_classes=self.config['model_params']['num_classes']
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Use focal loss
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                resnet_features = batch['resnet'].to(self.device)
                landmark_features = batch['landmarks'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Mixup augmentation
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(resnet_features.size(0))
                resnet_features = lam * resnet_features + (1 - lam) * resnet_features[index]
                landmark_features = lam * landmark_features + (1 - lam) * landmark_features[index]
                labels_a, labels_b = labels, labels[index]
                
                outputs = self.model(resnet_features, landmark_features)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_f1 = f1_score(train_labels, train_preds)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    resnet_features = batch['resnet'].to(self.device)
                    landmark_features = batch['landmarks'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(resnet_features, landmark_features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(test_loader)
            val_f1 = f1_score(val_labels, val_preds)
            
            # Log metrics
            logging.info(f'Epoch {epoch+1}:')
            logging.info(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
            logging.info(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
            
            # Save losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logging.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.model_results_dir / 'training_curves.png')
        
        return (X_resnet_test, X_landmarks_test), y_test
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Tuple of (resnet_features, landmark_features) to predict on
            
        Returns:
            array: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_resnet, X_landmarks = X
        
        # Scale features
        X_resnet = self.resnet_scaler.transform(X_resnet)
        X_landmarks = self.landmark_scaler.transform(X_landmarks)
        
        # Convert to tensor
        X_resnet = torch.FloatTensor(X_resnet).to(self.device)
        X_landmarks = torch.FloatTensor(X_landmarks).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_resnet, X_landmarks)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
        
    def predict_proba(self, X):
        """
        Make probability predictions.
        
        Args:
            X: Tuple of (resnet_features, landmark_features) to predict on
            
        Returns:
            array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_resnet, X_landmarks = X
        
        # Scale features
        X_resnet = self.resnet_scaler.transform(X_resnet)
        X_landmarks = self.landmark_scaler.transform(X_landmarks)
        
        # Convert to tensor
        X_resnet = torch.FloatTensor(X_resnet).to(self.device)
        X_landmarks = torch.FloatTensor(X_landmarks).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_resnet, X_landmarks)
            probs = torch.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
        
    def save_model(self):
        """Save the model and preprocessing components to disk."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Save model
        torch.save(self.model.state_dict(), self.model_dir / 'model.pt')
        logging.info(f"Model saved to {self.model_dir / 'model.pt'}")
        
        # Save scalers
        joblib.dump(self.resnet_scaler, self.model_dir / 'resnet_scaler.joblib')
        joblib.dump(self.landmark_scaler, self.model_dir / 'landmark_scaler.joblib')
        logging.info("Scalers saved")
        
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
        
        # Load scalers
        self.resnet_scaler = joblib.load(model_path / 'resnet_scaler.joblib')
        self.landmark_scaler = joblib.load(model_path / 'landmark_scaler.joblib')
        
        # Create and load model
        self.model = MultimodalFusionNN(
            resnet_dim=self.config['model_params']['resnet_dim'],
            landmark_dim=self.config['model_params']['landmark_dim'],
            num_classes=self.config['model_params']['num_classes']
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path / 'model.pt'))
        
        logging.info(f"Model and components loaded from {model_path}")

def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = MultimodalFusionModel()
    
    # Train model
    X_test, y_test = model.train()
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    # Visualize results
    model.visualize_results(results)
    
if __name__ == "__main__":
    main() 