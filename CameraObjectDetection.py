import cv2
import numpy as np
import json

#Pointer to webcam
video = cv2.VideoCapture(0)

#Counter
i = 0

while True:
    success, img = video.read()
    
    croppedImage = img[50:1870, 25:1055]
    
    greyscale = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

    ret, threshold = cv2.threshold(greyscale, 110, 255, cv2.THRESH_BINARY)
    
    kernel3x3 = np.ones((3, 3), np.uint8)
    
    erodeImage = cv2.erode(threshold, kernel3x3)
    
    cv2.imshow("webcam", erodeImage)

    i += 1
    cv2.waitKey(1)

