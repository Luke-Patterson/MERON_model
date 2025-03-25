# MERON Image Preprocessing

This module handles the preprocessing pipeline for MERON (Machine learning for Evaluation of Readiness for treating moderate and severe acute malnutrition Openly with MUAC and No weight and height), focusing on:

1. Child detection and cropping
2. Feature extraction using ResNet50
3. Creation of malnutrition status flags from metadata

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained model files:
   - Create a `preprocess_models` directory in this folder
   - Download the following files and place them in the `preprocess_models` directory:
     - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
     - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   - Download the `shape_predictor_68_face_landmarks.dat` file (for face landmark detection) and place it in the `data` directory

## Preprocessing Pipeline

The preprocessing pipeline consists of three main steps, to be executed in the following order:

### 1. Child Detection and Cropping (`run_cropping.py`)

This script processes raw images to detect and crop children:

```bash
python meron2/preprocessing/run_cropping.py
```

- Uses OpenCV-based child detection algorithm in `child_detection.py`
- Processes raw images from `data/raw_pictures/`
- Outputs cropped images to `data/cropped_pictures/`

### 2. Feature Extraction (`preprocess_resnet50.py`)

This script extracts ResNet50 features from cropped images:

```bash
python meron2/preprocessing/preprocess_resnet50.py
```

- Processes cropped images from `data/cropped_pictures/`
- Uses ResNet50 pre-trained model to extract 2048 image features
- Outputs feature vectors to `data/processed/resnet50_features.csv`

### 3. Malnutrition Flag Creation (`create_malnutrition_flags.py`)

This script processes metadata and calculates malnutrition flags:

```bash
python meron2/preprocessing/create_malnutrition_flags.py
```

- Uses WHO z-score tables in the `data` directory
- Processes metadata from `data/linkage_data/*.xlsx` files 
- Calculates WHZ scores and malnutrition status flags
- Creates two output files:
  - `data/processed/malnutrition_flags.csv`: Just the flags and metadata
  - `data/processed/features_with_flags.csv`: Combined ResNet50 features with malnutrition flags

## Output

The final output of the preprocessing pipeline is the `features_with_flags.csv` file containing:
- ResNet50 features (2048 dimensions)
- Photo ID and county information
- Anthropometric measurements (weight, height, MUAC)
- Calculated WHZ scores
- Malnutrition flags (SAM, MAM)

This dataset is ready to be used for training malnutrition classification models.

## Notes

- The detection model uses a confidence threshold of 0.5
- Cropped images include a 20% padding around the detected child
- If no child is detected, the original image is returned unchanged
- The WHO standards are used to calculate WHZ scores and malnutrition status 