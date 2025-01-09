import cv2
import numpy as np
from PIL import Image

class ImageEnhancer:
    @staticmethod
    def enhance_product(image):
        """Enhanced product image enhancement"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Color enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
        
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    @staticmethod
    def enhance_document(image):
        """Enhanced document image enhancement"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization using Otsu's method
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(threshold)
        
        # Improve contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        
        return Image.fromarray(enhanced)

    @staticmethod
    def enhance_landscape(image):
        """Enhanced landscape image enhancement"""
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Enhance brightness and contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        
        # Merge channels
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
        
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))