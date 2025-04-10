# MERON Model - Child Malnutrition Classification

This repository contains machine learning models for predicting child malnutrition from facial images. The project aims to develop accurate and efficient models that can classify children as either "Normal" or "Malnourished" (which includes both Moderate Acute Malnutrition (MAM) and Severe Acute Malnutrition (SAM)) based solely on facial images.

## Project Structure

```
data/
│   ├── raw_pictures/         # Original images
│   ├── cropped_pictures/     # Preprocessed images
│   ├── processed/            # Processed data for ML models 
│   ├── linkage_data/         # raw data for the original pictures
meron2/
├── modeling/
│   ├── experiments/          # Results and visualizations
│   │   └── model_comparison/ # Model comparison results
│   ├── models/              # Saved model files
│   ├── aanjankumar_resnet50_malnutrition.py
│   ├── advanced_ensemble_model.py
│   ├── landmark_model.py
│   ├── multimodal_fusion_model.py
│   ├── neural_network_model_20250327_improved.py
│   ├── vit_model.py
│   ├── xgboost_model.py
│   └── model_comparison.py   # Model comparison script
└── preprocessing/           # Image preprocessing scripts
```

## Available Models

1. **ResNet50 Model** (`aanjankumar_resnet50_malnutrition.py`)
   - Uses pre-trained ResNet50 architecture
   - Fine-tuned for malnutrition classification

2. **Advanced Ensemble Model** (`advanced_ensemble_model.py`)
   - Combines multiple models for improved performance
   - Uses weighted voting for final predictions

3. **Landmark Model** (`landmark_model.py`)
   - Uses facial landmarks for feature extraction
   - 68-point facial landmark detection

4. **Multimodal Fusion Model** (`multimodal_fusion_model.py`)
   - Combines multiple data modalities
   - Uses late fusion strategy

5. **Improved Neural Network** (`neural_network_model_20250327_improved.py`)
   - Custom neural network architecture
   - Optimized for malnutrition classification

6. **Vision Transformer** (`vit_model.py`)
   - Uses transformer architecture for image classification
   - State-of-the-art performance

7. **XGBoost Model** (`xgboost_model.py`)
   - Gradient boosting implementation
   - Uses extracted features from images

## Model Comparison

The `model_comparison.py` script provides a comprehensive comparison of all available models. It:

1. Runs each model with standardized evaluation
2. Collects performance metrics including:
   - Accuracy
   - F1 Score
   - Precision
   - Recall
   - ROC AUC
   - Training Time
   - Inference Time

3. Generates visualizations:
   - F1 Score comparison
   - Radar plot of key metrics
   - Training time comparison
   - ROC curves comparison

4. Saves results in:
   - CSV format (detailed metrics)
   - JSON format (summary statistics)
   - PNG format (visualizations)

### Running Model Comparison

```bash
python meron2/modeling/model_comparison.py
```

Results are saved in `meron2/modeling/experiments/model_comparison/[timestamp]/` with:
- `model_comparison.csv`: Detailed metrics for all models
- `summary.json`: Summary statistics
- `f1_scores.png`: F1 score comparison
- `radar_plot.png`: Multi-metric comparison
- `training_times.png`: Training time comparison
- `roc_curves.png`: ROC curves comparison

## Data

The models use images from:
- `data/raw_pictures/`: Original images
- `data/cropped_pictures/`: Preprocessed images

Target variables are derived from `malnutrition_flags.csv`:
- Binary classification: Normal (0) vs Malnourished (1)
- Malnourished includes both MAM and SAM cases

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- scikit-learn
- XGBoost
- OpenCV
- dlib
- pandas
- numpy
- matplotlib
- seaborn

## Usage

1. Ensure all dependencies are installed
2. Run the model comparison script:
   ```bash
   python meron2/modeling/model_comparison.py
   ```
3. View results in the experiments directory

## Results

The model comparison provides:
- Performance metrics for each model
- Visual comparisons of model performance
- ROC curves for binary classification
- Training and inference times
- Best performing model identification

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 