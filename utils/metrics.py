import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

class ImageEvaluator:
    @staticmethod
    def calculate_metrics(original_image, enhanced_image):
        """Calculate quality metrics between original and enhanced images"""
        # Convert PIL images to cv2 format
        orig = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        enhanced = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for SSIM
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate PSNR
        mse = np.mean((orig - enhanced) ** 2)
        if mse == 0:
            psnr = 100
        else:
            PIXEL_MAX = 255.0
            psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        
        # Calculate SSIM
        ssim_score = ssim(orig_gray, enhanced_gray)
        
        return {
            'psnr': psnr,
            'ssim': ssim_score
        }