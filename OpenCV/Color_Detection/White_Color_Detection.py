# Only Red color detection
import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White color
    low = np.array([0, 42, 0]) # lowest hue would be - 40,50,50( how do i found this i tested before and found this)
    high = np.array([179, 255, 255])
    #mask = cv2.inRange(hsv_frame, low_green, high_green)
     
    mask = cv2.inRange(hsv_frame, low, high) #we create maskk on hsv frame and then low green or high green
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Frame", frame)
    #cv2.imshow('Red mask', mask)
    cv2.imshow('Result', result)
 
    key = cv2.waitKey(1)
    if key ==27:
        break