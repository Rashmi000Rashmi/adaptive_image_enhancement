import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from configs.config import Config

class DomainClassifier:
    def __init__(self):
        print("Loading CLIP model...")
        self.device = Config.DEVICE
        self.model = CLIPModel.from_pretrained(Config.CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
        self.domains = Config.DOMAINS
        print("CLIP model loaded successfully!")

    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def classify(self, image_path):
        """Classify an image into one of the domains"""
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Prepare text inputs for each domain
        text_inputs = [f"This is a {domain} image" for domain in self.domains]
        
        # Process inputs
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Get image and text features
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.nn.functional.softmax(logits_per_image, dim=1)
        
        # Get predicted domain
        predicted_domain = self.domains[probs.argmax().item()]
        confidence = probs.max().item()
        
        return predicted_domain, confidence
