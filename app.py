import gradio as gr
from PIL import Image
import numpy as np
import cv2
from models.domain_classifier import DomainClassifier

def process_image(input_image):
    """Process the input image and return enhanced version"""
    if input_image is None:
        return None, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Save temporary image
        temp_path = "temp.jpg"
        input_image.save(temp_path)
        
        # Classify domain
        classifier = DomainClassifier()
        domain, confidence = classifier.classify(temp_path)
        
        # Convert to cv2 format for processing
        img_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
        # Apply enhancement based on domain
        if domain == 'product':
            enhanced = cv2.convertScaleAbs(img_cv, alpha=1.3, beta=0)
        elif domain == 'document':
            enhanced = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.adaptiveThreshold(enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return Image.fromarray(enhanced), f"Detected: {domain} (Confidence: {confidence:.2%})"
        else:  # landscape
            enhanced = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=10)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced_image = Image.fromarray(enhanced)
        
        return enhanced_image, f"Detected: {domain} (Confidence: {confidence:.2%})"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Enhanced Image"),
        gr.Textbox(label="Result")
    ],
    title="Adaptive Image Enhancement System",
    description="Upload an image to enhance it based on its content type."
)

if __name__ == "__main__":
    demo.launch()