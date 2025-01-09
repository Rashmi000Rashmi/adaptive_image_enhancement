# Adaptive Image Enhancement System

## Overview
This project is an AI-powered image enhancement system that automatically detects what type of image you have (like a product photo, a document scan, or a landscape picture) and applies the best enhancement techniques specifically designed for that type of image.

### What Problem Does It Solve?
- Not all images need the same type of enhancement
- A document needs different treatment than a landscape photo
- Manual image editing is time-consuming and requires expertise


## Technical Details

### Core Technologies
1. **CLIP Model for Image Classification**
   - Uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model
   - Helps understand the content and context of images
   - Can classify images into different domains (product/document/landscape)
   - Chosen for its robust understanding of diverse image types

2. **Image Processing Techniques**
   - **For Products:**
     - CLAHE (Contrast Limited Adaptive Histogram Equalization) for better color balance
     - Unsharp masking for detail enhancement
     - Selective color enhancement for product features
     - Denoising to maintain image quality

   - **For Documents:**
     - Otsu's thresholding for better text-background separation
     - Adaptive thresholding for handling uneven lighting
     - Text clarity enhancement
     - Grayscale optimization for readability

   - **For Landscapes:**
     - HSV color space manipulation for vibrant colors
     - Dynamic range adjustment for better sky/ground balance
     - Selective saturation enhancement
     - Edge-aware smoothing for natural looks

3. **Quality Metrics**
   - SSIM (Structural Similarity Index) for quality assessment
   - PSNR (Peak Signal-to-Noise Ratio) for noise evaluation
   - Brightness and contrast measurements
   - These metrics help validate enhancement quality

4. **Framework and Libraries**
   - PyTorch for deep learning operations
   - OpenCV for image processing
   - Gradio for user interface
   - Pillow for image handling

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps
1. Clone the repository:

- git clone https://github.com/Rashmi000Rashmi/adaptive_image_enhancement.git
- cd adaptive_image_enhancement

### Create and activate virtual environment:

### Windows
- python -m venv venv
- venv\Scripts\activate

### Linux/Mac
- python3 -m venv venv
- source venv/bin/activate

## Install required packages:

- pip install -r requirements.txt

## Run the application:

- python app.py

## Access the web interface:

- Open your browser
- Go to http://127.0.0.1:7860
- Upload an image and see the enhanced results

## Using the System

Upload any image through the web interface.
 The system automatically:

- Detects the image type
- Applies appropriate enhancements
- Provides quality metrics