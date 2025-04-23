import streamlit as st
import cv2
import torch
import numpy as np
import time
from model import load_model
from preprocess import preprocess_image
from hand_tracker import get_hand_image
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Load model
model = load_model()
model.eval()

# Streamlit UI
st.title("ASL Hand Sign Recognition & Typing")
st.write("Show a hand sign (A, B, C) to the camera. Press **Spacebar** to confirm the letter, **Backspace** to delete.")

# Initialize session state variables
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

if "last_detected_letter" not in st.session_state:
    st.session_state.last_detected_letter = ""

# Webcam feed placeholder
video_frame = st.empty()
letter_display = st.empty()

# OpenCV Webcam Setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        st.error("Error: Could not access webcam.")
        break

    hands, img = detector.findHands(img, draw=True)

    detected_letter = ""
    if hands:
        img_hand = get_hand_image(img)
        if img_hand is not None:
            img_tensor = preprocess_image(img_hand)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                detected_letter = ['A', 'B', 'C'][predicted.item()]
                st.session_state.last_detected_letter = detected_letter
                letter_display.markdown(f"### Detected Letter: **{detected_letter}**")

    # Convert frame for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video_frame.image(img_rgb, channels="RGB")

    # Display Typed Text with Unique Key
    st.text_area("Typed Text:", value=st.session_state.typed_text, height=100, key="typed_text")

    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Spacebar pressed → Confirm letter
        if st.session_state.last_detected_letter:
            st.session_state.typed_text += st.session_state.last_detected_letter
    elif key == 8:  # Backspace pressed → Delete last letter
        st.session_state.typed_text = st.session_state.typed_text[:-1]

cap.release()
cv2.destroyAllWindows()
