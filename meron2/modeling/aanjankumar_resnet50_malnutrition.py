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
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from PIL import Image

# Configuration
CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,  # 20% test set as per paper
    'VAL_SIZE': 0.2,   # 20% validation set as per paper
    'BATCH_SIZE': 16,  # Reduced batch size for better generalization
    'NUM_EPOCHS': 200,
    'LEARNING_RATE': 0.000001,  # Reduced learning rate for transfer learning
    'WEIGHT_DECAY': 0.01,  # Increased L2 regularization
    'PATIENCE': 20,
    'INPUT_SIZE': 224,
    'NUM_CLASSES': 2,
    'MOMENTUM': 0.9,
    'DROPOUT_RATE': 0.3,  # Reduced dropout
    'PREDICTION_THRESHOLD': 0.5,
    'UNFREEZE_LAYERS': 20  # Number of layers to unfreeze from the end
}

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class MalnutritionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MalnutritionClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        layers_to_unfreeze = [
            self.resnet.layer4,
            self.resnet.layer3[-1],
            self.resnet.layer3[-2]
        ]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Modify the final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x).squeeze()

class MalnutritionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.transform = transform or transforms.Compose([
            transforms.Resize((CONFIG['INPUT_SIZE'], CONFIG['INPUT_SIZE'])),
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
    """Load image paths and labels from CSV files"""
    # Load image paths and flags
    flags_df = pd.read_csv('data/processed/malnutrition_flags.csv')
    logging.info(f"Loaded {len(flags_df)} entries from malnutrition_flags.csv")
    
    # Get image paths
    image_dir = Path('data/cropped_pictures')
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
        # Check for exact match or match with any image extension
        matching_files = [f for f in available_files if f == photo_id or f.startswith(f"{photo_id}.")]
        if matching_files:
            # Take the first matching file
            image_paths.append(str(image_dir / matching_files[0]))
            valid_ids.append(photo_id)
        else:
            missing_ids.append(photo_id)
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir} or its subdirectories")
    
    logging.info(f"Found {len(image_paths)} images out of {len(flags_df)} entries")
    if missing_ids:
        logging.warning(f"Missing images for {len(missing_ids)} photo_ids")
        logging.warning(f"First 5 missing IDs: {missing_ids[:5]}")
    
    # Filter flags dataframe to only include valid images
    flags_df = flags_df[flags_df['photo_id'].isin(valid_ids)]
    
    # Sort both lists by photo_id to ensure alignment
    sorted_indices = np.argsort(valid_ids)
    image_paths = [image_paths[i] for i in sorted_indices]
    y = flags_df.sort_values('photo_id')['malnutrition'].values
    
    # Calculate class weights with balanced approach
    class_counts = np.bincount(y)
    if len(class_counts) == 0:
        raise ValueError("No valid labels found after filtering")
    
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    # Moderate weight increase for minority class
    minority_class = np.argmin(class_counts)
    class_weights[minority_class] *= 2  # Reduced from 4 to 2
    
    logging.info(f"Loaded {len(image_paths)} images")
    logging.info(f"Class distribution: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    
    return image_paths, y, class_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_dir, experiment_id):
    """Train the model with early stopping"""
    best_val_f1 = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > CONFIG['PREDICTION_THRESHOLD']).float()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='binary')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_outputs = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > CONFIG['PREDICTION_THRESHOLD']).float()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_outputs.extend(outputs.cpu().numpy())
        
        val_acc = 100 * accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary')
        
        # Log validation predictions distribution
        val_pred_dist = np.mean(val_preds)
        logging.info(f'Validation predictions distribution (percent positive): {val_pred_dist:.2%}')
        
        # Record metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'val_pred_dist': val_pred_dist
        }
        training_history.append(epoch_metrics)
        
        # Log metrics
        logging.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logging.info(f'Train Loss: {epoch_metrics["train_loss"]:.4f}, Train Acc: {epoch_metrics["train_acc"]:.2f}%, Train F1: {epoch_metrics["train_f1"]:.4f}')
        logging.info(f'Val Loss: {epoch_metrics["val_loss"]:.4f}, Val Acc: {epoch_metrics["val_acc"]:.2f}%, Val F1: {epoch_metrics["val_f1"]:.4f}')
        
        # Learning rate scheduling
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
    image_paths, y, class_weights = load_data()
    
    # First split: separate test set
    trainval_paths, test_paths, y_trainval, y_test = train_test_split(
        image_paths, y,
        test_size=CONFIG['TEST_SIZE'],
        random_state=CONFIG['RANDOM_STATE'],
        stratify=y
    )
    
    # Second split: separate validation set from training set
    train_paths, val_paths, y_train, y_val = train_test_split(
        trainval_paths, y_trainval,
        test_size=CONFIG['VAL_SIZE'],
        random_state=CONFIG['RANDOM_STATE'],
        stratify=y_trainval
    )
    
    logging.info(f"Split data into {len(train_paths)} training, {len(val_paths)} validation, and {len(test_paths)} test samples")
    
    # Create datasets
    train_dataset = MalnutritionDataset(train_paths, y_train)
    val_dataset = MalnutritionDataset(val_paths, y_val)
    test_dataset = MalnutritionDataset(test_paths, y_test)
    
    # Create data loaders with weighted sampling
    weights = [class_weights[label] for label in y_train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'])
    
    # Create model
    model = MalnutritionClassifier(num_classes=CONFIG['NUM_CLASSES']).to(DEVICE)
    
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