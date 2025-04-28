import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Dict, List
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
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

class MultimodalFusionModel(nn.Module):
    def __init__(self, resnet_dim: int, landmark_dim: int, num_classes: int = 2):
        super().__init__()
        
        # ResNet feature processing
        self.resnet_attention = AttentionLayer(resnet_dim)
        self.resnet_fc = nn.Sequential(
            nn.Linear(resnet_dim, 128),  # Reduced capacity
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Landmark feature processing
        self.landmark_attention = AttentionLayer(landmark_dim)
        self.landmark_fc = nn.Sequential(
            nn.Linear(landmark_dim, 64),  # Reduced capacity
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 32),  # Reduced capacity
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

class MultimodalTrainer:
    def __init__(self, 
                 resnet_features_path: str,
                 landmark_features_path: str,
                 labels_path: str,
                 output_dir: str = "model_output",
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 50,
                 patience: int = 10,
                 weight_decay: float = 1e-4):  # Added weight decay
        
        self.resnet_features_path = resnet_features_path
        self.landmark_features_path = landmark_features_path
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_data(self) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
        # Load features and labels
        resnet_features = pd.read_csv(self.resnet_features_path)
        landmark_features = pd.read_csv(self.landmark_features_path)
        labels = pd.read_csv(self.labels_path)
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
        
        # Split data
        X_resnet = resnet_features.drop('photo_id', axis=1).values
        X_landmarks = landmark_features.drop('photo_id', axis=1).values
        y = labels['malnutrition'].values
        
        # Split into train/val/test
        X_resnet_train, X_resnet_temp, X_landmarks_train, X_landmarks_temp, y_train, y_temp = train_test_split(
            X_resnet, X_landmarks, y, test_size=0.3, random_state=42, stratify=y
        )
        X_resnet_val, X_resnet_test, X_landmarks_val, X_landmarks_test, y_val, y_test = train_test_split(
            X_resnet_temp, X_landmarks_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Scale features
        scaler_resnet = StandardScaler()
        scaler_landmarks = StandardScaler()
        
        X_resnet_train = scaler_resnet.fit_transform(X_resnet_train)
        X_resnet_val = scaler_resnet.transform(X_resnet_val)
        X_resnet_test = scaler_resnet.transform(X_resnet_test)
        
        X_landmarks_train = scaler_landmarks.fit_transform(X_landmarks_train)
        X_landmarks_val = scaler_landmarks.transform(X_landmarks_val)
        X_landmarks_test = scaler_landmarks.transform(X_landmarks_test)
        
        # Create datasets
        train_dataset = MultimodalDataset(X_resnet_train, X_landmarks_train, y_train)
        val_dataset = MultimodalDataset(X_resnet_val, X_landmarks_val, y_val)
        test_dataset = MultimodalDataset(X_resnet_test, X_landmarks_test, y_test)
        
        # Create weighted samplers for class imbalance
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
            batch_size=self.batch_size,
            sampler=train_sampler
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }, {
            'train': (X_resnet_train, X_landmarks_train, y_train),
            'val': (X_resnet_val, X_landmarks_val, y_val),
            'test': (X_resnet_test, X_landmarks_test, y_test)
        }
    
    def train(self):
        # Load data
        dataloaders, data = self.load_data()
        
        # Initialize model
        model = MultimodalFusionModel(
            resnet_dim=data['train'][0].shape[1],
            landmark_dim=data['train'][1].shape[1]
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use focal loss
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{self.num_epochs}'):
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
                
                outputs = model(resnet_features, landmark_features)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(dataloaders['train'])
            train_f1 = f1_score(train_labels, train_preds)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in dataloaders['val']:
                    resnet_features = batch['resnet'].to(self.device)
                    landmark_features = batch['landmarks'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(resnet_features, landmark_features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(dataloaders['val'])
            val_f1 = f1_score(val_labels, val_preds)
            
            # Log metrics
            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
            
            # Save losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        
        return model
    
    def evaluate(self, model):
        # Load test data
        dataloaders, data = self.load_data()
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in dataloaders['test']:
                resnet_features = batch['resnet'].to(self.device)
                landmark_features = batch['landmarks'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(resnet_features, landmark_features)
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        f1 = f1_score(test_labels, test_preds)
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        
        # Log metrics
        logger.info('Test Results:')
        logger.info(f'F1 Score: {f1:.4f}')
        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        
        # Plot confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        return {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

def main():
    # Initialize trainer
    trainer = MultimodalTrainer(
        resnet_features_path='data/processed/resnet50_features.csv',
        landmark_features_path='data/processed/landmarks/landmark_features.csv',
        labels_path='data/processed/malnutrition_flags.csv',
        output_dir='meron2/modeling/experiments/multimodal_fusion',
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=50,
        patience=10,
        weight_decay=1e-4
    )
    
    # Train model
    model = trainer.train()
    
    # Evaluate model
    results = trainer.evaluate(model)
    
    # Save results
    with open(os.path.join(trainer.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main() 