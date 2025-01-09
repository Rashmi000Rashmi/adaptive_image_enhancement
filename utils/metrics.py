import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class ImageQualityMetrics:
    @staticmethod
    def calculate_metrics(original, enhanced):
        """Calculate image quality metrics"""
        # Convert images to grayscale for SSIM
        original_gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
        enhanced_gray = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(original_gray, enhanced_gray)
        
        # Calculate PSNR
        psnr_score = psnr(original_gray, enhanced_gray)
        
        # Calculate Brightness
        brightness_original = np.mean(original_gray)
        brightness_enhanced = np.mean(enhanced_gray)
        brightness_change = ((brightness_enhanced - brightness_original) / brightness_original) * 100
        
        return {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'brightness_change': brightness_change
        }