"""
Neural Network Model for Malnutrition Classification - Improved Version

Incorporates:
1. Dimensionality reduction with PCA
2. More aggressive class balancing
3. Higher class weights for minority classes
4. Feature selection
5. Additional regularization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import json
from datetime import datetime
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import math
import torch.nn.functional as F

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COLS_START = 1  # The ResNet50 features start from column 1
FEATURE_COLS_END = 2049  # The ResNet50 features end at column 2048
RESULTS_DIR = os.path.join('meron2', 'modeling', 'results')
MODELS_DIR = os.path.join('meron2', 'modeling', 'models')
DATA_PATH = os.path.join('data', 'processed', 'features_with_flags.csv')
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 25
VALIDATION_SPLIT = 0.2
PCA_COMPONENTS = 300  # Number of principal components to keep
BALANCE_ALPHA = 0.9  # Aggressive balance between natural and equal distribution

# Ensure the output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class MalnutritionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MalnutritionNN(nn.Module):
    def __init__(self, input_size, num_classes):
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

def create_malnutrition_class(row):
    """
    Create a three-class target variable:
    - 0: Normal
    - 1: MAM (Moderate Acute Malnutrition)
    - 2: SAM (Severe Acute Malnutrition)
    """
    if row['sam'] == 1:
        return 2
    elif row['mam'] == 1:
        return 1
    else:
        return 0

def load_and_preprocess_data():
    """
    Load and preprocess the data with dimensionality reduction
    """
    start_time = time.time()
    print("\n=== Data Loading and Preprocessing ===")
    
    print("Loading data from CSV...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows of data")
    
    # Create the three-class target variable
    print("\nCreating three-class target variable...")
    df['malnutrition_class'] = df.apply(create_malnutrition_class, axis=1)
    
    # Get the class distribution
    class_distribution = df['malnutrition_class'].value_counts().sort_index()
    print("\nClass distribution:")
    for class_name, count in zip(['Normal', 'MAM', 'SAM'], class_distribution):
        print(f"{class_name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Select features (ResNet50 features)
    print("\nExtracting ResNet50 features...")
    X = df.iloc[:, FEATURE_COLS_START:FEATURE_COLS_END].values
    print(f"Original feature shape: {X.shape}")
    
    # Select target
    y = df['malnutrition_class'].values
    
    # Split into train and test sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Perform feature selection first to remove irrelevant features
    print("\nPerforming feature selection...")
    feature_selector = SelectKBest(f_classif, k=500)
    X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = feature_selector.transform(X_test_scaled)
    print(f"Feature shape after selection: {X_train_selected.shape}")
    
    # Apply PCA for dimensionality reduction
    print(f"\nApplying PCA to reduce dimensions to {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance after PCA: {explained_variance:.2f}%")
    print(f"Feature shape after PCA: {X_train_pca.shape}")
    
    print(f"\nTraining set shape: {X_train_pca.shape}")
    print(f"Test set shape: {X_test_pca.shape}")
    
    duration = time.time() - start_time
    print(f"\nData preprocessing completed in {duration:.2f} seconds")
    
    return X_train_pca, X_test_pca, y_train, y_test, class_distribution

def create_balanced_sampler(labels, alpha=BALANCE_ALPHA):
    """
    Create a weighted sampler that balances between natural distribution and complete balance
    Alpha controls how much we balance (0 = natural distribution, 1 = completely balanced)
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Natural distribution
    natural_dist = class_counts / total_samples
    
    # Equal distribution
    equal_dist = np.ones_like(class_counts) / len(class_counts)
    
    # Blend distributions with alpha
    target_dist = (1 - alpha) * natural_dist + alpha * equal_dist
    
    print(f"Target class distribution: {target_dist}")
    
    # Calculate weights for each sample
    weights = np.zeros_like(labels, dtype=np.float32)
    for t in range(len(class_counts)):
        weights[labels == t] = target_dist[t] / class_counts[t]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, l1_weight, num_epochs):
    """
    Train the neural network model with L1 regularization
    """
    print("\n=== Model Training ===")
    start_time = time.time()
    
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"Training on {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"L1 regularization weight: {l1_weight}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, l1_reg = model(features)
            loss = criterion(outputs, labels) + l1_weight * l1_reg
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels for F1 calculation
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"\rEpoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}", end="")
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs, l1_reg = model(features)
                loss = criterion(outputs, labels) + l1_weight * l1_reg
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels for F1 calculation
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print predictions distribution and per-class metrics
        val_pred_counts = np.bincount(val_preds, minlength=3)
        val_pred_percentages = val_pred_counts / len(val_preds) * 100
        
        # Per-class F1 scores
        val_f1_per_class = f1_score(val_labels, val_preds, average=None, zero_division=0)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"\rEpoch [{epoch+1}/{num_epochs}] LR: {current_lr:.6f} "
              f"Train Loss: {train_loss:.4f} Train F1: {train_f1:.4f} "
              f"Val Loss: {val_loss:.4f} Val F1: {val_f1:.4f} "
              f"Time: {epoch_duration:.2f}s")
        
        print(f"Prediction distribution: Normal: {val_pred_percentages[0]:.1f}%, "
              f"MAM: {val_pred_percentages[1]:.1f}%, SAM: {val_pred_percentages[2]:.1f}%")
        
        print(f"F1 per class: Normal: {val_f1_per_class[0]:.4f}, "
              f"MAM: {val_f1_per_class[1]:.4f}, SAM: {val_f1_per_class[2]:.4f}")
        
        # Early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            print(f"New best model with validation F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.2f} seconds")
    print(f"Best model saved from epoch {best_epoch}")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    
    return best_model_state

def evaluate_model(model, test_loader, device, l1_weight=0.0):
    """
    Evaluate the model on the test set
    """
    print("\n=== Model Evaluation ===")
    start_time = time.time()
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs, l1_reg = model(features)
            loss = criterion(outputs, labels) + l1_weight * l1_reg
            test_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"\rProcessed {batch_idx + 1}/{len(test_loader)} batches", end="")
    
    test_loss /= len(test_loader)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    
    print("\nCalculating metrics...")
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Class-specific metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Calculate per-class accuracy for each class
    conf_matrix = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    balanced_accuracy = np.mean(per_class_accuracy)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, 
                                      target_names=['Normal', 'MAM', 'SAM'], 
                                      output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Prediction distribution
    pred_counts = np.bincount(all_preds, minlength=3)
    pred_percentages = pred_counts / len(all_preds) * 100
    print(f"Test prediction distribution: Normal: {pred_percentages[0]:.1f}%, "
          f"MAM: {pred_percentages[1]:.1f}%, SAM: {pred_percentages[2]:.1f}%")
    
    # ROC AUC score (one-vs-rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC score: {e}")
        roc_auc = None
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'roc_auc': roc_auc,
        'test_loss': test_loss
    }
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(['Normal', 'MAM', 'SAM']):
        print(f"{class_name}: Accuracy={per_class_accuracy[i]:.4f}, F1={f1_per_class[i]:.4f}, "
              f"Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}")
    
    duration = time.time() - start_time
    print(f"\nEvaluation completed in {duration:.2f} seconds")
    
    return results

def save_model_and_results(model, results, timestamp):
    """
    Save the model and results to disk
    """
    # Save model
    model_path = os.path.join(MODELS_DIR, f'neural_network_model_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f'neural_network_results_{timestamp}.json')
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results_serializable[k] = v.tolist()
        elif isinstance(v, np.integer):
            results_serializable[k] = int(v)
        elif isinstance(v, np.floating):
            results_serializable[k] = float(v)
        else:
            results_serializable[k] = v
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Create and save visualizations
    visualize_results(results, timestamp)
    
    return model_path, results_path

def visualize_results(results, timestamp):
    """
    Create and save visualizations of the results
    """
    # Create confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    conf_matrix = np.array(results['confusion_matrix'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'MAM', 'SAM'],
                yticklabels=['Normal', 'MAM', 'SAM'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{timestamp}.png'))
    
    # Create bar chart of performance metrics per class
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame({
        'Precision': results['precision_per_class'],
        'Recall': results['recall_per_class'],
        'F1 Score': results['f1_per_class'],
        'Accuracy': results['per_class_accuracy']
    }, index=['Normal', 'MAM', 'SAM'])
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Performance Metrics per Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'performance_metrics_{timestamp}.png'))
    
    plt.close('all')

def main():
    """
    Main function
    """
    total_start_time = time.time()
    print("Neural Network Model for Malnutrition Classification - Improved Version")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load and preprocess data with dimensionality reduction
    X_train, X_test, y_train, y_test, class_distribution = load_and_preprocess_data()
    
    # Split training data to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"Training set shape after validation split: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = MalnutritionDataset(X_train, y_train)
    val_dataset = MalnutritionDataset(X_val, y_val)
    test_dataset = MalnutritionDataset(X_test, y_test)
    
    # Create a more aggressively balanced sampler (alpha=0.9)
    print(f"\nCreating aggressively balanced sampler (alpha={BALANCE_ALPHA})...")
    train_sampler = create_balanced_sampler(y_train, alpha=BALANCE_ALPHA)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\nInitializing model...")
    input_size = X_train.shape[1]  # PCA reduced features
    num_classes = len(class_distribution)
    model = MalnutritionNN(input_size, num_classes).to(device)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Higher class weights - more emphasis on minority classes
    class_weights = torch.FloatTensor([1.0, 3.0, 5.0]).to(device)
    print(f"\nStrong class weights: {class_weights}")
    
    # Create loss function
    print("\nSetting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use Adam with optimized parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6  # Reduced as we're using explicit L1 regularization
    )
    
    # Learning rate scheduler - reduce on plateau based on F1 score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # L1 regularization weight
    l1_weight = 1e-5
    
    # Print optimization objective
    print("\nOptimization approach: Dimensionality reduction + Aggressive class balancing")
    print(f"Using PCA to reduce to {PCA_COMPONENTS} dimensions")
    print(f"Using stronger class weights [1.0, 3.0, 5.0] and sampler alpha={BALANCE_ALPHA}")
    print(f"Using L1 regularization with weight {l1_weight}")
    
    # Train model with scheduler and L1 regularization
    best_model_state = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, l1_weight, NUM_EPOCHS)
    
    # Load best model state
    print("\nLoading best model state...")
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, l1_weight)
    
    # Generate timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and results
    print("\nSaving model and results...")
    model_path, results_path = save_model_and_results(model, results, timestamp)
    
    total_duration = time.time() - total_start_time
    print("\n" + "=" * 50)
    print(f"Total execution time: {total_duration:.2f} seconds")
    print(f"Neural network model training and evaluation completed!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main() 