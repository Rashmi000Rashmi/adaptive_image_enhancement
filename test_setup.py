import torch
import transformers
import diffusers
import cv2
import wandb

def test_imports():
    print("Testing imports:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Diffusers version: {diffusers.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test PyTorch
    x = torch.rand(2, 3)
    print("\nTest PyTorch tensor:")
    print(x)

if __name__ == "__main__":
    test_imports()