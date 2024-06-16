import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import matplotlib.pyplot as plt
import math

# Load the trained model
model_path = 'vgg16_hand_gesture_model.h5'
model = load_model(model_path)

# Create a dictionary for the labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C'
    # Add all labels if needed
}

# Initialize the hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

offset = 20
imgSize = 224


def predict_gesture(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    return labels_dict[predicted_label_index]


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        predicted_label = predict_gesture(imgWhite)
        cv2.rectangle(img, (x-offset, y-offset),
                      (x+w+offset, y+h+offset), (0, 255, 0), 2)
        cv2.putText(img, predicted_label, (x, y-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
