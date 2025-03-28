import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import optuna
from collections import Counter
import torch.nn.functional as F
import numpy as np
import joblib
from datetime import datetime

# Configuration
CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'BATCH_SIZE': 32,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 1e-5,
    'L1_LAMBDA': 1e-5,
    'FEATURE_SELECTION_K': 1000,  # Initial feature selection
    'PCA_COMPONENTS': 300,        # After feature selection
    'N_SPLITS': 5,                # Number of cross-validation folds
    'PATIENCE': 10,               # Early stopping patience
    'CLASS_WEIGHTS': [1.0, 3.0, 5.0],
    'USE_SMOTE': True,
    'SMOTE_RATIO': 0.8,           # 0.5 = halfway between original and balanced
    'N_TRIALS': 20,               # Number of Bayesian optimization trials
    'N_MODELS': 5,                # Number of models in ensemble
}

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class MalnutritionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MalnutritionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.4):
        super(MalnutritionNN, self).__init__()
        self.input_size = input_size
        
        # Create list of sequential layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            linear = nn.Linear(prev_size, hidden_size)
            
            # Initialize weights using Kaiming initialization
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.constant_(linear.bias, 0)
            
            layers.append(linear)
            
            # Add batch norm, activation, and dropout after each linear layer except the last
            if i < len(hidden_sizes) - 1:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg

def load_data(resnet_features_path, labels_path):
    """
    Load and preprocess the data from CSV files
    """
    print(f"Loading features from: {resnet_features_path}")
    
    # Load features from CSV
    df = pd.read_csv(resnet_features_path)
    print(f"Loaded data with {len(df)} rows")
    
    # Check if we need to create the target variable
    if 'malnutrition_class' not in df.columns:
        print("Creating malnutrition class from 'mam' and 'sam' flags...")
        # Create the three-class target variable as in neural_network_model_20250327.py
        df['malnutrition_class'] = df.apply(create_malnutrition_class, axis=1)
    
    # Select features (ResNet50 features start from column 1 and end at column 2048)
    feature_cols = list(range(1, 2049))
    resnet_features = df.iloc[:, feature_cols].values
    
    # Select target
    y = df['malnutrition_class'].values
    
    # Print class distribution
    class_counts = np.bincount(y)
    print("\nClass distribution:")
    for class_idx, count in enumerate(['Normal', 'MAM', 'SAM']):
        if class_idx < len(class_counts):
            print(f"{count}: {class_counts[class_idx]} samples ({class_counts[class_idx]/len(y)*100:.1f}%)")
    
    print(f"Data loaded: {resnet_features.shape[0]} samples with {resnet_features.shape[1]} features")
    print(f"Class distribution: {Counter(y)}")
    
    return resnet_features, y

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

def preprocess_features(features, labels, config, fit_preprocessing=True, preprocessing=None):
    """Preprocess features with feature selection and PCA"""
    if fit_preprocessing:
        # Feature selection using ANOVA F-value
        selector = SelectKBest(f_classif, k=config['FEATURE_SELECTION_K'])
        features_selected = selector.fit_transform(features, labels)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_selected)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=config['PCA_COMPONENTS'], random_state=config['RANDOM_STATE'])
        features_pca = pca.fit_transform(features_scaled)
        
        # Save preprocessing objects
        preprocessing = {
            'selector': selector,
            'scaler': scaler,
            'pca': pca
        }
        
        return features_pca, preprocessing
    else:
        # Apply saved preprocessing steps
        if preprocessing is None:
            preprocessing = config.get('preprocessing')
            if preprocessing is None:
                raise ValueError("No preprocessing objects provided. Either fit_preprocessing must be True, preprocessing must be provided directly, or config must contain 'preprocessing' key.")
            
        features_selected = preprocessing['selector'].transform(features)
        features_scaled = preprocessing['scaler'].transform(features_selected)
        features_pca = preprocessing['pca'].transform(features_scaled)
        return features_pca, preprocessing

def apply_smote(X_train, y_train, config):
    """Apply SMOTE for oversampling minority classes"""
    if config['USE_SMOTE']:
        # Calculate target ratios for SMOTE
        class_counts = Counter(y_train)
        max_class_count = max(class_counts.values())
        target_ratio = {
            class_label: int(max_class_count * config['SMOTE_RATIO']) 
            if count < max_class_count else count
            for class_label, count in class_counts.items()
        }
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=target_ratio, random_state=config['RANDOM_STATE'])
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Before SMOTE: {Counter(y_train)}")
        print(f"After SMOTE: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    else:
        return X_train, y_train

def create_balanced_sampler(labels, alpha=0.9):
    """Create a partially balanced sampler with alpha controlling the balance degree"""
    # Calculate class weights for balanced sampling
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    
    # Calculate weights for each sample
    weights = np.ones_like(labels, dtype=np.float64)
    
    # Blend between natural distribution (alpha=0) and equal distribution (alpha=1)
    for class_idx in range(len(class_counts)):
        natural_weight = n_samples / class_counts[class_idx] if class_counts[class_idx] > 0 else 0
        equal_weight = len(class_counts)  # All classes equally likely
        blended_weight = (1 - alpha) * 1.0 + alpha * natural_weight
        weights[labels == class_idx] = blended_weight
    
    # Create and return sampler
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def objective(trial, X_train, y_train, X_val, y_val, input_size, config):
    """Objective function for Optuna hyperparameter optimization"""
    # Define hyperparameters to optimize
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 1e-4, log=True)
    
    # Create model with suggested hyperparameters
    model = MalnutritionNN(
        input_size=input_size,
        hidden_sizes=[hidden_size1, hidden_size2, 3],
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    # Create dataset and data loader
    train_dataset = MalnutritionDataset(X_train, y_train)
    sampler = create_balanced_sampler(y_train, alpha=0.9)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        sampler=sampler
    )
    
    # Define loss function with class weights
    class_weights = torch.FloatTensor(config['CLASS_WEIGHTS']).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(features)
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
        val_dataset = MalnutritionDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])
        val_preds = []
        val_labels = []
        
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
        
        # Calculate F1 score (macro)
        f1 = f1_score(val_labels, val_preds, average='macro')
    
    return f1

def train_and_evaluate_model(X_train, y_train, X_val, y_val, best_params, config):
    """Train and evaluate a model with the given parameters"""
    # Create model with best hyperparameters
    model = MalnutritionNN(
        input_size=X_train.shape[1],
        hidden_sizes=[best_params['hidden_size1'], best_params['hidden_size2'], 3],
        dropout_rate=best_params['dropout_rate']
    ).to(DEVICE)
    
    # Create datasets and data loaders
    train_dataset = MalnutritionDataset(X_train, y_train)
    val_dataset = MalnutritionDataset(X_val, y_val)
    
    sampler = create_balanced_sampler(y_train, alpha=0.9)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['BATCH_SIZE']
    )
    
    # Define loss function with class weights
    class_weights = torch.FloatTensor(config['CLASS_WEIGHTS']).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['NUM_EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Add L1 regularization
            l1_reg = model.l1_regularization()
            loss = loss + best_params['l1_lambda'] * l1_reg
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_f1_per_class = f1_score(train_labels, train_preds, average=None)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Store predictions for metrics
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_f1_per_class = f1_score(val_labels, val_preds, average=None)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Train F1 per class: {train_f1_per_class}")
        print(f"Val F1 per class: {val_f1_per_class}")
        print(f"Class distribution in predictions: {Counter(val_preds)}")
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['PATIENCE']:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_probs = []
        
        for features, labels in val_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            val_preds.extend(predicted.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
    
    return model, val_preds, val_probs, best_f1

def create_ensemble_model(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Create and train an ensemble of models using cross-validation"""
    # Get directories from main config
    MODELS_DIR = os.path.join('meron2', 'modeling', 'models')
    
    results = {
        'config': {k: v for k, v in config.items() if k != 'preprocessing'},  # Don't store preprocessing objects
        'models': [],
        'predictions': [],
        'fold_scores': [],
        'test_metrics': {}
    }
    
    # Apply SMOTE for data augmentation on the training set
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, config)
    
    # Create cross-validation folds
    kf = StratifiedKFold(n_splits=config['N_SPLITS'], shuffle=True, random_state=config['RANDOM_STATE'])
    
    fold_predictions = []
    fold_probabilities = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_resampled, y_train_resampled)):
        print(f"\n*** Training model for fold {fold+1}/{config['N_SPLITS']} ***")
        
        # Split data for this fold
        X_fold_train, y_fold_train = X_train_resampled[train_idx], y_train_resampled[train_idx]
        X_fold_val, y_fold_val = X_train_resampled[val_idx], y_train_resampled[val_idx]
        
        # Hyperparameter optimization using Optuna
        print(f"Running Bayesian hyperparameter optimization with {config['N_TRIALS']} trials...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(
            lambda trial: objective(trial, X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                                  X_train.shape[1], config),
            n_trials=config['N_TRIALS']
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        print(f"Best hyperparameters for fold {fold+1}: {best_params}")
        
        # Train model with best hyperparameters
        model, val_preds, val_probs, val_f1 = train_and_evaluate_model(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val, best_params, config
        )
        
        # Save model and results
        model_filename = os.path.join(MODELS_DIR, f"ensemble_model_fold_{fold+1}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
        
        # Store predictions on validation data
        fold_predictions.append(val_preds)
        fold_probabilities.append(val_probs)
        
        # Store fold score
        results['fold_scores'].append(val_f1)
        
        # Store model info
        results['models'].append({
            'fold': fold + 1,
            'hyperparameters': best_params,
            'validation_f1': val_f1,
            'model_file': model_filename
        })
        
        print(f"Fold {fold+1} completed with validation F1 score: {val_f1:.4f}")

    # Generate ensemble predictions for test set
    all_test_predictions = []
    all_test_probabilities = []
    
    for fold in range(config['N_SPLITS']):
        # Load model
        model = MalnutritionNN(
            input_size=X_test.shape[1],
            hidden_sizes=[
                results['models'][fold]['hyperparameters']['hidden_size1'],
                results['models'][fold]['hyperparameters']['hidden_size2'],
                3
            ],
            dropout_rate=results['models'][fold]['hyperparameters']['dropout_rate']
        ).to(DEVICE)
        model.load_state_dict(torch.load(results['models'][fold]['model_file']))
        
        # Generate predictions
        model.eval()
        test_dataset = MalnutritionDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'])
        test_preds = []
        test_probs = []
        
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(DEVICE)
                outputs = model(features)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        all_test_predictions.append(test_preds)
        all_test_probabilities.append(test_probs)
    
    # Ensemble predictions (soft voting)
    ensemble_probabilities = np.mean(all_test_probabilities, axis=0)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    
    # Calculate ensemble metrics
    test_acc = accuracy_score(y_test, ensemble_predictions)
    test_f1_macro = f1_score(y_test, ensemble_predictions, average='macro')
    test_f1_per_class = f1_score(y_test, ensemble_predictions, average=None)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, ensemble_predictions)
    
    # Save metrics
    results['test_metrics'] = {
        'accuracy': test_acc,
        'f1_macro': test_f1_macro,
        'f1_per_class': test_f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions_distribution': Counter(ensemble_predictions),
        'ground_truth_distribution': Counter(y_test)
    }
    
    # Generate classification report
    report = classification_report(y_test, ensemble_predictions, zero_division=0)
    print("\nEnsemble Classification Report:\n", report)
    
    # Print confusion matrix
    print("\nEnsemble Confusion Matrix:")
    print(cm)
    
    return results

def visualize_results(results):
    """Visualize model results"""
    # Get results directory
    RESULTS_DIR = os.path.join('meron2', 'modeling', 'results')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Confusion Matrix
    cm = np.array(results['test_metrics']['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'MAM', 'SAM'],
                yticklabels=['Normal', 'MAM', 'SAM'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f'ensemble_confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # F1 Scores
    plt.figure(figsize=(10, 6))
    f1_scores = results['test_metrics']['f1_per_class']
    bars = plt.bar(['Normal', 'MAM', 'SAM'], f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Class')
    plt.ylim(0, 1)
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    f1_path = os.path.join(RESULTS_DIR, f'ensemble_f1_scores_{timestamp}.png')
    plt.savefig(f1_path)
    plt.tight_layout()
    print(f"F1 scores plot saved to {f1_path}")
    
    # Distribution of predictions
    plt.figure(figsize=(10, 6))
    pred_dist = results['test_metrics']['predictions_distribution']
    plt.bar(['Normal', 'MAM', 'SAM'], 
            [pred_dist.get(0, 0), pred_dist.get(1, 0), pred_dist.get(2, 0)])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, f'ensemble_prediction_distribution_{timestamp}.png')
    plt.savefig(dist_path)
    print(f"Prediction distribution plot saved to {dist_path}")
    
    # Cross-validation F1 scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(results['fold_scores'])+1), results['fold_scores'], marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Validation F1 Score')
    plt.title('Cross-Validation F1 Scores')
    plt.grid(True)
    plt.tight_layout()
    cv_path = os.path.join(RESULTS_DIR, f'ensemble_cv_f1_scores_{timestamp}.png')
    plt.savefig(cv_path)
    print(f"Cross-validation F1 scores plot saved to {cv_path}")
    
    # Close all plots
    plt.close('all')

def save_results(results, filename=None):
    """Save results to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filename}")

def main():
    # Start timing
    start_time = time.time()
    
    # Setup directories for results
    RESULTS_DIR = os.path.join('meron2', 'modeling', 'results')
    MODELS_DIR = os.path.join('meron2', 'modeling', 'models')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data with same path as neural_network_model
    DATA_PATH = os.path.join('data', 'processed', 'features_with_flags.csv')
    print(f"Loading data from {DATA_PATH}")
    
    # Load data
    X, y = load_data(DATA_PATH, None)  # We don't need a separate labels path
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['RANDOM_STATE'],
        stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Apply feature selection and PCA for dimensionality reduction
    print("Applying feature selection and dimensionality reduction...")
    X_train_processed, preprocessing = preprocess_features(X_train, y_train, CONFIG)
    X_test_processed, _ = preprocess_features(X_test, y_test, CONFIG, fit_preprocessing=False, preprocessing=preprocessing)
    
    print(f"Processed training set: {X_train_processed.shape}")
    print(f"Processed test set: {X_test_processed.shape}")
    
    # Save preprocessing objects
    preprocessing_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')
    joblib.dump(preprocessing, preprocessing_path)
    print(f"Preprocessing pipeline saved to {preprocessing_path}")
    
    # Update config with preprocessing objects
    CONFIG['preprocessing'] = preprocessing
    
    # Create and train ensemble model
    print("\n=== Training Ensemble Model ===")
    results = create_ensemble_model(
        X_train_processed, y_train,
        X_train_processed, y_train,  # Using same data for validation in cross-validation
        X_test_processed, y_test,
        CONFIG
    )
    
    # Visualize results
    print("\n=== Visualizing Results ===")
    visualize_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f'ensemble_results_{timestamp}.json')
    save_results(results, results_path)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Print key metrics
    print("\n=== Final Results ===")
    print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Macro F1 Score: {results['test_metrics']['f1_macro']:.4f}")
    print("F1 Scores by Class:")
    for i, class_name in enumerate(['Normal', 'MAM', 'SAM']):
        print(f"  {class_name}: {results['test_metrics']['f1_per_class'][i]:.4f}")

if __name__ == "__main__":
    main() 