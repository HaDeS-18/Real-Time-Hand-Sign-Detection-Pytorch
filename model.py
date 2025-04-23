import torch
import torch.nn as nn
import cv2
import numpy as np
class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes=3):  # Adjust based on your dataset
        super(SignLanguageClassifier, self).__init__()
        self.fc1 = nn.Linear(90000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Final layer for classification

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_model(model_path="hand_sign_model.pth", num_classes=3):
    """Loads the trained model from a .pth file."""
    model = SignLanguageClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image_for_model(img, final_size=300):
    """Converts image to grayscale, resizes to 300x300, and flattens into a 90,000-column row."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (final_size, final_size), interpolation=cv2.INTER_LINEAR)
    img_flattened = img_resized.flatten().astype(np.float32) / 255.0  # Normalize
    return torch.tensor(img_flattened).unsqueeze(0)  # Convert to tensor and add batch dimension
