import os
from models.domain_classifier import DomainClassifier
from utils.metrics import ImageEvaluator
import cv2
from PIL import Image
import numpy as np

class ImageEnhancer:
    def __init__(self):
        self.domain_classifier = DomainClassifier()
        
    def enhance_product(self, image):
        """Enhance product images"""
        # Convert PIL image to cv2 format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic enhancement for products
        # Increase contrast
        contrast = 1.3
        brightness = 0
        enhanced = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    def enhance_document(self, image):
        """Enhance document images"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(enhanced)
    
    def enhance_landscape(self, image):
        """Enhance landscape images"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Increase saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=0)
        
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    def enhance_image(self, image_path):
        """Main enhancement function"""
        # Load image
        image = Image.open(image_path)
        
        # Classify domain
        domain, confidence = self.domain_classifier.classify(image_path)
        
        print(f"Detected domain: {domain} (confidence: {confidence:.2%})")
        
        # Apply domain-specific enhancement
        if domain == 'product':
            enhanced = self.enhance_product(image)
        elif domain == 'document':
            enhanced = self.enhance_document(image)
        elif domain == 'landscape':
            enhanced = self.enhance_landscape(image)
        
        return enhanced, domain

def test_enhancement():
    enhancer = ImageEnhancer()
    evaluator = ImageEvaluator()
    
    # Process all images in test_images folder
    test_folder = "test_images"
    output_folder = "enhanced_images"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image
    for image_file in os.listdir(test_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, image_file)
            print(f"\nProcessing {image_file}...")
            
            try:
                # Get enhanced image
                enhanced_image, domain = enhancer.enhance_image(image_path)
        
                # Calculate metrics
                original_image = Image.open(image_path)
                metrics = evaluator.calculate_metrics(original_image, enhanced_image)
        
                print(f"Enhancement Metrics:")
                print(f"PSNR: {metrics['psnr']:.2f} dB")
                print(f"SSIM: {metrics['ssim']:.4f}")
        
                # Save enhanced image
                output_path = os.path.join(output_folder, f"enhanced_{domain}_{image_file}")
                enhanced_image.save(output_path)
        
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    test_enhancement()