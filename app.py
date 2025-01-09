import gradio as gr
from PIL import Image
import numpy as np
import os
from models.domain_classifier import DomainClassifier
from utils.enhancement import ImageEnhancer

def process_image(input_image):
    """Process the input image with advanced enhancements"""
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
        
        # Apply domain-specific enhancement
        if domain == 'product':
            enhanced_image = ImageEnhancer.enhance_product(input_image)
            details = "Enhanced product details, sharpness, and color accuracy"
        elif domain == 'document':
            enhanced_image = ImageEnhancer.enhance_document(input_image)
            details = "Improved readability and text clarity"
        else:  # landscape
            enhanced_image = ImageEnhancer.enhance_landscape(input_image)
            details = "Enhanced colors, contrast, and natural features"
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Save enhanced image
        output_path = os.path.join("output_images", f"enhanced_{domain}.jpg")
        enhanced_image.save(output_path)
        
        result_text = f"""
        Detected: {domain} (Confidence: {confidence:.2%})
        Enhancements Applied: {details}
        Enhanced image saved to: {output_path}
        """
        
        return enhanced_image, result_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image")
    ],
    outputs=[
        gr.Image(type="pil", label="Enhanced Image"),
        gr.Textbox(label="Analysis Results", lines=3)
    ],
    title="Advanced Adaptive Image Enhancement System",
    description="""
    Upload any image to automatically enhance it based on its content type.
    The system detects whether it's a product, document, or landscape image
    and applies appropriate enhancements.
    """,
)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("output_images"):
        os.makedirs("output_images")
    demo.launch()