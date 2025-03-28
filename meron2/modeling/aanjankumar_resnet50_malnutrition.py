import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configuration
CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,  # 20% test set as per paper
    'VAL_SIZE': 0.2,   # 20% validation set as per paper
    'BATCH_SIZE': 32,  # Reduced batch size for better generalization
    'NUM_EPOCHS': 200, # Increased epochs to allow for better convergence
    'LEARNING_RATE': 0.00001,  # Slightly increased learning rate
    'WEIGHT_DECAY': 0.0001,  # L2 regularization
    'PATIENCE': 20,  # Increased patience for early stopping
    'INPUT_SIZE': 224,  # Standard ResNet50 input size
    'NUM_CLASSES': 2,  # Binary classification
    'MOMENTUM': 0.9,  # Momentum for SGD
    'DROPOUT_RATE': 0.5,  # Dropout rate for regularization
    'PREDICTION_THRESHOLD': 0.5  # Balanced threshold
}

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class MalnutritionClassifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=2):
        super(MalnutritionClassifier, self).__init__()
        
        # Fully connected layers for processing pre-extracted ResNet50 features
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Reduced dropout
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Reduced dropout
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()

class MalnutritionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)  # Features are already 2048-dimensional
        self.labels = torch.FloatTensor(labels)  # Binary classification
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def setup_experiment():
    """Setup experiment directories and logging"""
    # Create timestamp and experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"resnet50_malnutrition_{timestamp}"
    
    # Create experiment directory structure
    base_dir = Path('meron2/modeling/experiments')
    experiment_dir = base_dir / experiment_id
    model_dir = experiment_dir / 'models'
    results_dir = experiment_dir / 'results'
    logs_dir = experiment_dir / 'logs'
    
    # Create directories
    for dir_path in [model_dir, results_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f'training_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save configuration
    with open(experiment_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    return experiment_dir, model_dir, results_dir, logs_dir

def load_data():
    """Load features and labels from CSV files"""
    # Load features and flags
    features_df = pd.read_csv('data/processed/resnet50_features_aanjankumar.csv')
    flags_df = pd.read_csv('data/processed/malnutrition_flags.csv')
    
    # Get the first column name from features_df (should be photo_id)
    photo_id_col = features_df.columns[0]
    
    # Ensure we have matching samples
    common_ids = set(features_df[photo_id_col]) & set(flags_df['photo_id'])
    features_df = features_df[features_df[photo_id_col].isin(common_ids)]
    flags_df = flags_df[flags_df['photo_id'].isin(common_ids)]
    
    # Sort both dataframes by photo_id to ensure alignment
    features_df = features_df.sort_values(photo_id_col)
    flags_df = flags_df.sort_values('photo_id')
    
    # Remove photo_id column from features
    X = features_df.drop(columns=[photo_id_col]).values
    y = flags_df['malnutrition'].values
    
    # Calculate class weights with balanced approach
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    # Moderate weight increase for minority class
    minority_class = np.argmin(class_counts)
    class_weights[minority_class] *= 2  # Reduced from 4 to 2
    
    logging.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logging.info(f"Class distribution: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    
    return X, y, class_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_dir, experiment_id):
    """Train the model with early stopping"""
    best_val_f1 = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > CONFIG['PREDICTION_THRESHOLD']).float()  # Updated threshold
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        train_f1 = f1_score(train_labels, train_preds, average='binary')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > CONFIG['PREDICTION_THRESHOLD']).float()  # Updated threshold
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
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
        logging.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logging.info(f'Train Loss: {epoch_metrics["train_loss"]:.4f}, Train Acc: {epoch_metrics["train_acc"]:.2f}%, Train F1: {epoch_metrics["train_f1"]:.4f}')
        logging.info(f'Val Loss: {epoch_metrics["val_loss"]:.4f}, Val Acc: {epoch_metrics["val_acc"]:.2f}%, Val F1: {epoch_metrics["val_f1"]:.4f}')
        
        # Learning rate scheduling - pass validation loss to scheduler
        scheduler.step(val_loss/len(val_loader))
        
        # Early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best model
            model_path = model_dir / f'best_model_{experiment_id}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'training_history': training_history
            }, model_path)
            logging.info(f'Saved best model to {model_path}')
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                logging.info(f'Early stopping after {epoch+1} epochs')
                break
    
    return best_val_f1, training_history

def evaluate_model(model, test_loader):
    """Evaluate the model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store raw probabilities
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            probs = outputs.cpu().numpy()  # Get raw probabilities
            predicted = (outputs > CONFIG['PREDICTION_THRESHOLD']).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    return accuracy, f1, cm, report, all_probs, all_labels

def visualize_results(cm, accuracy, f1, training_history, results_dir, experiment_id, probs, labels):
    """Visualize model results"""
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malnourished'],
                yticklabels=['Normal', 'Malnourished'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(results_dir / f'confusion_matrix_{experiment_id}.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot([m['epoch'] for m in training_history], [m['train_loss'] for m in training_history], label='Train Loss')
    plt.plot([m['epoch'] for m in training_history], [m['val_loss'] for m in training_history], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot([m['epoch'] for m in training_history], [m['train_acc'] for m in training_history], label='Train Acc')
    plt.plot([m['epoch'] for m in training_history], [m['val_acc'] for m in training_history], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / f'training_history_{experiment_id}.png')
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / f'roc_curve_{experiment_id}.png')
    
    # Save metrics
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'training_history': training_history
    }
    
    with open(results_dir / f'metrics_{experiment_id}.json', 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Start timing
    start_time = time.time()
    
    # Setup experiment
    experiment_dir, model_dir, results_dir, logs_dir = setup_experiment()
    experiment_id = experiment_dir.name
    
    logging.info(f"Starting experiment: {experiment_id}")
    logging.info(f"Configuration: {CONFIG}")
    
    # Load data
    X, y, class_weights = load_data()
    
    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=CONFIG['TEST_SIZE'],
        random_state=CONFIG['RANDOM_STATE'],
        stratify=y
    )
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=CONFIG['VAL_SIZE'],
        random_state=CONFIG['RANDOM_STATE'],
        stratify=y_trainval
    )
    
    logging.info(f"Split data into {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
    
    # Create datasets
    train_dataset = MalnutritionDataset(X_train, y_train)
    val_dataset = MalnutritionDataset(X_val, y_val)
    test_dataset = MalnutritionDataset(X_test, y_test)
    
    # Create data loaders with weighted sampling
    weights = [class_weights[label] for label in y_train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'])
    
    # Create model
    model = MalnutritionClassifier(input_size=X.shape[1], num_classes=CONFIG['NUM_CLASSES']).to(DEVICE)
    
    # Define loss function with class weights
    pos_weight = torch.tensor([class_weights[1]], device=DEVICE)  # Weight for positive class
    criterion = nn.BCELoss()  # Changed to BCELoss since we're using sigmoid
    
    # Define optimizer with Adam and adjusted learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler with ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train model
    logging.info("\n=== Training Model ===")
    best_val_f1, training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        CONFIG['NUM_EPOCHS'], model_dir, experiment_id
    )
    
    # Load best model
    checkpoint = torch.load(model_dir / f'best_model_{experiment_id}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logging.info("\n=== Evaluating Model ===")
    accuracy, f1, cm, report, probs, labels = evaluate_model(model, test_loader)
    
    # Print detailed results
    logging.info(f"\nTest Accuracy: {accuracy:.4f}")
    logging.info(f"Test F1 Score: {f1:.4f}")
    logging.info("\nClassification Report:")
    logging.info(report)
    
    # Print confusion matrix
    logging.info("\nConfusion Matrix:")
    logging.info("[[TN FP]")
    logging.info(" [FN TP]]")
    logging.info(f"[[{cm[0,0]} {cm[0,1]}]")
    logging.info(f" [{cm[1,0]} {cm[1,1]}]]")
    
    # Visualize results
    logging.info("\n=== Visualizing Results ===")
    visualize_results(cm, accuracy, f1, training_history, results_dir, experiment_id, probs, labels)
    
    # Calculate total time
    total_time = time.time() - start_time
    logging.info(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    logging.info(f"\nExperiment completed. Results saved in: {experiment_dir}")

if __name__ == "__main__":
    main() 