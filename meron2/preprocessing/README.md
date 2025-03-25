# MERON Image Preprocessing

This module handles the preprocessing of MERON images, specifically focusing on child detection and cropping.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained model files:
   - Create a `models` directory in this folder
   - Download the following files and place them in the `models` directory:
     - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
     - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

## Usage

The preprocessing pipeline can be used to process individual images or entire directories:

```python
from child_detection import process_image, process_directory

# Process a single image
process_image('input_image.jpg', 'output_image.jpg')

# Process an entire directory
process_directory('input_directory', 'output_directory')
```

## Features

1. **Child Detection**: Uses a pre-trained deep learning model to detect children in images
2. **Smart Cropping**: Crops the image to focus on the detected child with appropriate padding
3. **Batch Processing**: Can process entire directories of images while maintaining the directory structure

## Notes

- The detection model uses a confidence threshold of 0.5
- Cropped images include a 20% padding around the detected child
- If no child is detected, the original image is returned unchanged
- The pipeline preserves the original image format and quality 