import torch

class Config:
    # Model settings
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    IMAGE_SIZE = 224  # image size
    
    # Training settings
    BATCH_SIZE = 4  # Small batch size for CPU
    
   
    DOMAINS = ['product', 'document', 'landscape']
    
    DEVICE = "cpu"
