"""
XGBoost Model for Malnutrition Classification

This script trains an XGBoost model to classify children into three classes:
- Normal: No malnutrition
- MAM: Moderate Acute Malnutrition
- SAM: Severe Acute Malnutrition

The model uses ResNet50 features extracted from cropped child images.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COLS_START = 1  # The ResNet50 features start from column 1
FEATURE_COLS_END = 2049  # The ResNet50 features end at column 2048
RESULTS_DIR = os.path.join('meron2', 'modeling', 'results')
MODELS_DIR = os.path.join('meron2', 'modeling', 'models')
DATA_PATH = os.path.join('data', 'processed', 'features_with_flags.csv')

# Ensure the output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
    Load and preprocess the data:
    - Load the features_with_flags.csv file
    - Create the three-class target variable
    - Split into features and target
    - Split into train and test sets
    """
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Create the three-class target variable
    print("Creating three-class target variable...")
    df['malnutrition_class'] = df.apply(create_malnutrition_class, axis=1)
    
    # Get the class distribution
    class_distribution = df['malnutrition_class'].value_counts().sort_index()
    print(f"Class distribution:\n{class_distribution}")
    
    # Select features (ResNet50 features)
    X = df.iloc[:, FEATURE_COLS_START:FEATURE_COLS_END].values
    
    # Select target
    y = df['malnutrition_class'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, class_distribution

def train_model(X_train, y_train, class_distribution):
    """
    Train the XGBoost model with grid search for hyperparameter tuning
    """
    print("Training XGBoost model...")
    
    # Calculate class weights based on class distribution
    total_samples = class_distribution.sum()
    class_weights = {
        i: total_samples / (len(class_distribution) * count) 
        for i, count in enumerate(class_distribution)
    }
    print(f"Class weights: {class_weights}")
    
    # Set up the XGBoost classifier with class weights
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    # Apply class weights to the samples
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    # Simple parameter grid for demonstration
    param_grid = {
        'n_estimators': [1000, 2000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search with 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro',  # Use macro-averaged F1 score for imbalanced classes
        n_jobs=-1,
        verbose=3  # Increased verbosity to show more detailed progress
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    # Class-specific metrics
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=['Normal', 'MAM', 'SAM'], output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC AUC score (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except:
        roc_auc = None
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'roc_auc': roc_auc
    }
    
    return results

def save_model_and_results(model, results, best_params):
    """
    Save the model and results to disk
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'xgboost_model_{timestamp}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f'xgboost_results_{timestamp}.json')
    
    # Add best parameters to results
    results['best_params'] = best_params
    
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
        'F1 Score': results['f1_per_class']
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
    print("XGBoost Model for Malnutrition Classification")
    print("=" * 50)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, class_distribution = load_and_preprocess_data()
    
    # Train model
    model, best_params = train_model(X_train, y_train, class_distribution)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Save model and results
    model_path, results_path = save_model_and_results(model, results, best_params)
    
    print("=" * 50)
    print(f"XGBoost model training and evaluation completed!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main() 