import gradio as gr
from PIL import Image
import numpy as np
import os
import json
import datetime
from models.domain_classifier import DomainClassifier
from utils.enhancement import ImageEnhancer

# Create necessary directories
DIRS = [
    'data/training',
    'data/test',
    'experiments/enhancement_tests',
    'experiments/metrics',
    'flagged',
    'logs'
]

for dir_path in DIRS:
    os.makedirs(dir_path, exist_ok=True)

def save_experiment(original_image, enhanced_image, domain, metrics):
    """Save experiment results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save images
    exp_path = f"experiments/enhancement_tests/{timestamp}"
    os.makedirs(exp_path, exist_ok=True)
    
    original_path = f"{exp_path}/original.jpg"
    enhanced_path = f"{exp_path}/enhanced_{domain}.jpg"
    
    original_image.save(original_path)
    enhanced_image.save(enhanced_path)
    
    # Save metrics
    metrics_path = f"experiments/metrics/{timestamp}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return exp_path

def process_image(input_image, enable_logging=True):
    """Process the input image with advanced enhancements"""
    if input_image is None:
        return None, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # Save to test data directory
        test_path = os.path.join("data/test", f"test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        input_image.save(test_path)
        
        # Classify domain
        classifier = DomainClassifier()
        domain, confidence = classifier.classify(test_path)
        
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
        
        # Calculate metrics and save experiment
        metrics = {
            "domain": domain,
            "confidence": float(confidence),
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details
        }
        
        if enable_logging:
            exp_path = save_experiment(input_image, enhanced_image, domain, metrics)
            
            # Log processing details
            log_path = os.path.join("logs", f"processing_{datetime.datetime.now().strftime('%Y%m%d')}.log")
            with open(log_path, 'a') as f:
                f.write(f"\n{datetime.datetime.now().isoformat()}: Processed image {test_path} -> {exp_path}")
        
        result_text = f"""
        Detected: {domain} (Confidence: {confidence:.2%})
        Enhancements Applied: {details}
        Results saved in experiments folder
        """
        
        return enhanced_image, result_text
        
    except Exception as e:
        # Log error
        with open(os.path.join("logs", "errors.log"), 'a') as f:
            f.write(f"\n{datetime.datetime.now().isoformat()}: Error - {str(e)}")
        return None, f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
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
    flagging_options=["Good Enhancement", "Poor Enhancement", "Wrong Classification"],
    allow_flagging="manual"
)

if __name__ == "__main__":
    demo.launch()