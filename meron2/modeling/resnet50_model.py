import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from PIL import Image
from base_model import BaseModel

class MalnutritionDataset(Dataset):
    """Dataset class for malnutrition image data."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        # Reshape labels to [N, 1] shape
        self.labels = torch.FloatTensor(labels).view(-1, 1)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class ResNet50Model(BaseModel):
    """ResNet50 model for malnutrition classification."""
    
    def __init__(self, config=None):
        """
        Initialize the ResNet50 model.
        
        Args:
            config (dict): Model configuration parameters
        """
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'val_size': 0.2,
            'batch_size': 16,
            'num_epochs': 200,
            'learning_rate': 0.000001,
            'weight_decay': 0.01,
            'patience': 20,
            'input_size': 224,
            'num_classes': 2,
            'momentum': 0.9,
            'dropout_rate': 0.3,
            'prediction_threshold': 0.5,
            'unfreeze_layers': 20,
            'data_path': os.path.join('data', 'processed', 'malnutrition_flags.csv'),
            'image_dir': os.path.join('data', 'cropped_pictures')
        }
        
        # Merge default config with provided config
        if config:
            default_config.update(config)
            
        super().__init__('resnet50', default_config)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
    def _create_model(self):
        """Create and configure the ResNet50 model."""
        # Load pre-trained ResNet50
        model = models.resnet50(pretrained=True)
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        layers_to_unfreeze = [
            model.layer4,
            model.layer3[-1],
            model.layer3[-2]
        ]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Modify the final layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.config['dropout_rate']),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.config['dropout_rate']),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        return model.to(self.device)
        
    def load_data(self):
        """
        Load and preprocess the data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, class_distribution)
        """
        logging.info("Loading data...")
        flags_df = pd.read_csv(self.config['data_path'])
        
        # Get image paths
        image_dir = Path(self.config['image_dir'])
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory {image_dir} does not exist")
        
        image_paths = []
        valid_ids = []
        missing_ids = []
        
        # Get list of files in directory
        available_files = [f.name for f in image_dir.iterdir() if f.is_file()]
        logging.info(f"Found {len(available_files)} files in directory")
        
        # Match photo_ids with files
        for photo_id in flags_df['photo_id']:
            matching_files = [f for f in available_files if f == photo_id or f.startswith(f"{photo_id}.")]
            if matching_files:
                image_paths.append(str(image_dir / matching_files[0]))
                valid_ids.append(photo_id)
            else:
                missing_ids.append(photo_id)
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        
        logging.info(f"Found {len(image_paths)} images out of {len(flags_df)} entries")
        if missing_ids:
            logging.warning(f"Missing images for {len(missing_ids)} photo_ids")
        
        # Filter flags dataframe to only include valid images
        flags_df = flags_df[flags_df['photo_id'].isin(valid_ids)]
        
        # Sort both lists by photo_id to ensure alignment
        sorted_indices = np.argsort(valid_ids)
        image_paths = [image_paths[i] for i in sorted_indices]
        y = flags_df.sort_values('photo_id')['malnutrition'].values
        
        # Calculate class weights
        class_counts = np.bincount(y)
        total_samples = len(y)
        class_weights = total_samples / (len(class_counts) * class_counts)
        minority_class = np.argmin(class_counts)
        class_weights[minority_class] *= 2
        
        logging.info(f"Class distribution: {class_counts}")
        logging.info(f"Class weights: {class_weights}")
        
        # First split: separate test set
        trainval_paths, test_paths, y_trainval, y_test = train_test_split(
            image_paths, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Second split: separate validation set from training set
        train_paths, val_paths, y_train, y_val = train_test_split(
            trainval_paths, y_trainval,
            test_size=self.config['val_size'],
            random_state=self.config['random_state'],
            stratify=y_trainval
        )
        
        # Create datasets
        train_dataset = MalnutritionDataset(train_paths, y_train)
        val_dataset = MalnutritionDataset(val_paths, y_val)
        test_dataset = MalnutritionDataset(test_paths, y_test)
        
        # Create data loaders with weighted sampling
        weights = [class_weights[label] for label in y_train]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        
        return train_loader, val_loader, test_loader, class_weights
        
    def train(self):
        """
        Train the ResNet50 model.
        
        Returns:
            tuple: (test_loader, y_test) for evaluation
        """
        # Load data
        train_loader, val_loader, test_loader, class_weights = self.load_data()
        
        # Create model
        self.model = self._create_model()
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                # Reshape labels to match output shape
                labels = labels.view(-1, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > self.config['prediction_threshold']).float()
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_acc = 100 * accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='binary')
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > self.config['prediction_threshold']).float()
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='binary')
            
            # Record metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss/len(train_loader),
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss/len(val_loader),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_metrics)
            
            # Log metrics
            logging.info(f'Epoch [{epoch+1}/{self.config["num_epochs"]}]')
            logging.info(f'Train Loss: {epoch_metrics["train_loss"]:.4f}, Train Acc: {epoch_metrics["train_acc"]:.2f}%, Train F1: {epoch_metrics["train_f1"]:.4f}')
            logging.info(f'Val Loss: {epoch_metrics["val_loss"]:.4f}, Val Acc: {epoch_metrics["val_acc"]:.2f}%, Val F1: {epoch_metrics["val_f1"]:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss/len(val_loader))
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logging.info(f'Early stopping after {epoch+1} epochs')
                    break
        
        # Get test labels
        y_test = []
        for _, labels in test_loader:
            y_test.extend(labels.cpu().numpy())
        y_test = np.array(y_test)
        
        return test_loader, y_test
        
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
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.to(self.device))
            predicted = (outputs > self.config['prediction_threshold']).float()
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
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.to(self.device))
            return outputs.cpu().numpy()
        
    def save_model(self):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Save model state
        model_path = self.model_dir / 'model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, model_path)
        logging.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path):
        """
        Load the model from disk.
        
        Args:
            model_path (str): Path to the model directory
        """
        model_path = Path(model_path)
        
        # Load model state
        checkpoint = torch.load(model_path / 'model.pth')
        self.config = checkpoint['config']
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {model_path / 'model.pth'}")
        
def main():
    """Main function to run the model training and evaluation pipeline."""
    # Initialize model
    model = ResNet50Model()
    
    # Train model
    test_loader, y_test = model.train()
    
    # Save model
    model.save_model()
    
    # Evaluate model
    results = model.evaluate(test_loader, y_test)
    
    # Visualize results
    model.visualize_results(results)
    
if __name__ == "__main__":
    main() 