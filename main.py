import cv2
import numpy as np
import torch
from model import load_model, preprocess_image_for_model  # Correct import
from cvzone.HandTrackingModule import HandDetector
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the model
model = load_model()

# Modify the output layer to handle only 3 classes (A, B, C)
num_classes = 3
model.fc3 = torch.nn.Linear(256, num_classes)

# Set the model to evaluation mode
model.eval()

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        print("Error: Couldn't capture image from webcam.")
        continue  # Skip this frame

    hands, img = detector.findHands(img,draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure coordinates are within the image bounds
        hImg, wImg, _ = img.shape
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, wImg), min(y + h + offset, hImg)

        imgCrop = img[y1:y2, x1:x2]

        # Ensure cropping didn't fail
        if imgCrop.size == 0:
            print("Warning: Cropped image is empty!")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Preprocess the image for model input (flatten and normalize)
        img_to_predict = preprocess_image_for_model(imgWhite)

        # Feed the image into the model for prediction
        with torch.no_grad():
            outputs = model(img_to_predict)
            _, predicted = torch.max(outputs, 1)
            predicted_label = ['A', 'B', 'C'][predicted.item()]  # Mapping output to 'A', 'B', or 'C'

        # Display the prediction on the image
        cv2.putText(imgWhite, f'Predicted: {predicted_label}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the cropped and resized image with prediction
        cv2.imshow("Predicted Image", imgWhite)

    cv2.imshow("Webcam Feed", img)
    cv2.waitKey(1)
