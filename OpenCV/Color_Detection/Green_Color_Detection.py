# Only Red color detection
import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green color
    low_green = np.array([40, 50, 50]) # lowest hue would be - 40,50,50( how do i found this i tested before and found this)
    high_green = np.array([80, 255, 255])
    #mask = cv2.inRange(hsv_frame, low_green, high_green)
     
    green_mask = cv2.inRange(hsv_frame, low_green, high_green) #we create maskk on hsv frame and then low green or high green
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    cv2.imshow("Frame", frame)
    #cv2.imshow('Red mask', mask)
    cv2.imshow('Green', green)
 
    key = cv2.waitKey(1)
    if key ==27:
        break