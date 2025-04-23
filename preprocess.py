import cv2
import numpy as np
import torch

def preprocess_image(img, final_size=300, device=None):
    """Converts image to grayscale, resizes to 300x300, and flattens into a 90,000-column row."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    img_flattened = img_resized.flatten().astype(np.float32) / 255.0  # Normalize
    
    # Create tensor and move to the appropriate device
    tensor_img = torch.tensor(img_flattened).unsqueeze(0)  # Add batch dimension
    if device:
        tensor_img = tensor_img.to(device)  # Move tensor to GPU if available
    
    return tensor_img
