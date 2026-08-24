# Only Red color detection
import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blue color
    low_blue = np.array([110, 50, 50]) # lowest hue would be - 110,50,50( how do i found this i tested before and found this)
    high_blue = np.array([130, 255, 255])
    #mask = cv2.inRange(hsv_frame, low_blue, high_blue)
     
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue) #we create maskk on hsv frame and then low blue or high blue
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    cv2.imshow("Frame", frame)
    #cv2.imshow('Red mask', mask)
    cv2.imshow('Blue', blue)
 
    key = cv2.waitKey(1)
    if key ==27:
        break