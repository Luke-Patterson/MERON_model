"""
Image Dimension Analysis Tool

This script provides utilities to analyze image dimensions in different directories:
1. check_image_dimensions(): Lists all image dimensions in raw_pictures directory
2. analyze_image_dimensions(): Performs statistical analysis of image dimensions in cropped_pictures directory

The analysis includes:
- Total number of images
- Number of unique dimensions
- Statistical measures (mean, median, standard deviation) for both width and height
- Minimum and maximum dimensions
- Most common dimensions and their frequencies

Usage:
    python check_dimensions.py
"""

from PIL import Image
import os
import numpy as np
from collections import Counter
from datetime import datetime

def setup_logging():
    """Setup logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'image_dimension_analysis_{timestamp}.log'
    
    def log_message(message):
        print(message)  # Console output
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    return log_message

def check_image_dimensions():
    log = setup_logging()
    path = 'data/raw_pictures'
    sizes = set()
    
    log(f"\n=== Checking Raw Pictures Dimensions ===\n")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    filepath = os.path.join(root, filename)
                    with Image.open(filepath) as img:
                        sizes.add(img.size)
                        log(f"Image {filepath}: {img.size}")
                except Exception as e:
                    log(f"Error processing {filepath}: {e}")
    
    log(f"\nUnique dimensions found: {sizes}\n")
    log("=" * 50)

def analyze_cropped_image_dimensions():
    log = setup_logging()
    path = 'data/cropped_pictures'
    dimensions = []
    sizes = set()
    
    log(f"\n=== Analyzing Cropped Pictures Dimensions ===\n")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    filepath = os.path.join(root, filename)
                    with Image.open(filepath) as img:
                        width, height = img.size
                        dimensions.append((width, height))
                        sizes.add((width, height))
                except Exception as e:
                    log(f"Error processing {filepath}: {e}")
    
    if not dimensions:
        log("No images found in the specified directory")
        return
    
    # Convert dimensions to numpy arrays for analysis
    widths = np.array([d[0] for d in dimensions])
    heights = np.array([d[1] for d in dimensions])
    
    # Calculate statistics
    log("\n=== Image Dimension Analysis ===")
    log(f"Total number of images: {len(dimensions)}")
    log(f"Number of unique dimensions: {len(sizes)}")
    
    log("\nWidth Statistics:")
    log(f"  Mean: {np.mean(widths):.2f}")
    log(f"  Median: {np.median(widths):.2f}")
    log(f"  Standard Deviation: {np.std(widths):.2f}")
    log(f"  Min: {np.min(widths)}")
    log(f"  Max: {np.max(widths)}")
    
    log("\nHeight Statistics:")
    log(f"  Mean: {np.mean(heights):.2f}")
    log(f"  Median: {np.median(heights):.2f}")
    log(f"  Standard Deviation: {np.std(heights):.2f}")
    log(f"  Min: {np.min(heights)}")
    log(f"  Max: {np.max(heights)}")
    
    # Most common dimensions
    dimension_counts = Counter(dimensions)
    log("\nMost Common Dimensions:")
    for (width, height), count in dimension_counts.most_common(5):
        log(f"  {width}x{height}: {count} images")
    
    log("\n" + "=" * 50)

if __name__ == "__main__":
    # print("Checking raw pictures dimensions:")
    # check_image_dimensions()
    print("\nAnalyzing cropped pictures dimensions:")
    analyze_cropped_image_dimensions() 