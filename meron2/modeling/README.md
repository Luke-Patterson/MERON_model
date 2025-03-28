# MERON Modeling

This directory contains the machine learning models for the MERON (Machine learning for Evaluation of Readiness for treating moderate and severe acute malnutrition Openly with MUAC and No weight and height) project.

## Models

### XGBoost Classifier (`xgboost_model.py`)

A multi-class classifier that predicts malnutrition status as one of three categories:
- Normal (0): No malnutrition
- MAM (1): Moderate Acute Malnutrition
- SAM (2): Severe Acute Malnutrition

Features:
- Uses ResNet50 features extracted from cropped child images
- Handles class imbalance using class weights
- Performs hyperparameter tuning using grid search with cross-validation
- Evaluates performance using various metrics including F1 score, precision, and recall

### Directory Structure

- `models/`: Contains saved model files (serialized using joblib)
- `results/`: Contains evaluation results (JSON) and visualizations

## Usage

To train the XGBoost model:

```bash
python meron2/modeling/xgboost_model.py
```

This will:
1. Load the processed features from `data/processed/features_with_flags.csv`
2. Create a three-class target variable
3. Split the data into training and test sets (80/20 split)
4. Train the XGBoost model with hyperparameter tuning
5. Evaluate the model on the test set
6. Save the model and results to the appropriate directories

## Model Evaluation

The model is evaluated using the following metrics:
- Accuracy
- Precision (per class and macro-averaged)
- Recall (per class and macro-averaged)
- F1 Score (per class and macro-averaged)
- Confusion Matrix
- ROC AUC (One vs Rest)

## Class Imbalance

The dataset has class imbalance, with SAM cases being only ~3.7% of the data and MAM cases ~16.1%. This imbalance is addressed by:
1. Using class weights during training
2. Evaluating with F1 score (macro-averaged) instead of accuracy
3. Using stratified sampling for train/test splits

## Future Work

Potential improvements to explore:
1. Ensemble methods combining multiple models
2. Deep learning approaches with a custom neural network on top of ResNet50 features
3. Feature selection or dimensionality reduction
4. More advanced handling of class imbalance (e.g., SMOTE)
5. Adding additional features from metadata 