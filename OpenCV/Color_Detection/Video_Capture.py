# Capture_Video.py

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
    
    
    

    
    