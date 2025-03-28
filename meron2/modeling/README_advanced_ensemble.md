# Advanced Ensemble Model for Malnutrition Classification

This implementation combines multiple advanced techniques to maximize model performance with limited training data:

## Key Features

### 1. Cross-Validation (K-Fold)
- Uses 5-fold cross-validation to get more reliable performance estimates
- Each fold trains a separate model that becomes part of the final ensemble
- Provides robust evaluation metrics across different data splits

### 2. Data Augmentation (SMOTE)
- Applies Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples for minority classes
- Configurable balance ratio (0.8 default) to create a more balanced dataset without going to full equality
- Helps model learn patterns in underrepresented classes (MAM and SAM)

### 3. Feature Selection & Dimensionality Reduction
- Two-step dimensionality reduction pipeline:
  1. SelectKBest with ANOVA F-test to select the most relevant 1000 features
  2. PCA to further reduce to 300 principal components
- Removes noise while preserving the most important signal patterns
- Critical for preventing overfitting with limited data

### 4. Bayesian Hyperparameter Optimization
- Uses Optuna for efficient hyperparameter tuning
- Optimizes:
  - Hidden layer sizes
  - Dropout rates
  - Learning rate
  - Weight decay (L2 regularization)
  - L1 regularization strength
- Requires fewer trials than grid search to find optimal settings

### 5. Model Ensembling
- Trains models on different cross-validation folds
- Uses soft voting (averaging class probabilities) to make final predictions
- Reduces variance and improves generalization

### 6. Advanced Regularization
- Combines multiple regularization techniques:
  - L1 regularization for feature sparsity
  - L2 regularization (weight decay) for smaller weights
  - Dropout for network robustness
  - Batch normalization for stable training
  - Early stopping based on validation F1 score
  - Learning rate reduction on plateau

## Model Architecture
- Optimized architecture using Bayesian search
- Two hidden layers with sizes determined by optimization
- Kaiming initialization for better gradient flow
- Batch normalization and dropout after each layer

## Class Balancing Strategies
- Class-weighted loss function: [1.0, 3.0, 5.0] for [Normal, MAM, SAM]
- Weighted random sampling during training with alpha=0.9
- SMOTE for synthetic minority samples

## Running the Model
```
python meron2/modeling/advanced_ensemble_model.py
```

## Output
- Trained model weights for each fold
- Comprehensive metrics including:
  - Per-class F1 scores
  - Confusion matrix
  - Classification report
  - Cross-validation scores
- Visualization plots saved as PNG files
- Full results saved in timestamped JSON file

## Configuration
Key parameters can be adjusted in the CONFIG dictionary at the top of the script:
- Feature selection and PCA dimensions
- SMOTE ratio and usage flag
- Number of trials for Bayesian optimization
- Cross-validation folds
- Early stopping patience
- Class weights 