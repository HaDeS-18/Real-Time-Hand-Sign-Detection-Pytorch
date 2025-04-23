import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

def get_hand_image(img):
    """
    Detects a hand in the image and returns a cropped 300x300 image with a white background.
    Returns None if no hand is detected.
    """
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure coordinates are within the image bounds
        hImg, wImg, _ = img.shape
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, wImg), min(y + h + offset, hImg)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return None

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

        return imgWhite  # Cropped hand image

    return None  # No hand detected
