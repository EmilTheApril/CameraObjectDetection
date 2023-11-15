import cv2
import numpy as np
import json

#Pointer to webcam
video = cv2.VideoCapture(0)

#Counter
i = 0

while True:
    #Turns on camera and saves the image in img
    success, img = video.read()
    
    #Crops the image, to fit the whiteboard
    croppedImage = img[50:1870, 25:1055]
    
    #Grayscales the image
    greyscale = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

    #Makes the whole image black and white
    ret, threshold = cv2.threshold(greyscale, 110, 255, cv2.THRESH_BINARY)
    
    #Kernel used to enlargen all pixels
    kernel3x3 = np.ones((3, 3), np.uint8)
    
    #Enlargens all pixels
    erodeImage = cv2.erode(threshold, kernel3x3)
    
    #Shows the image/video
    cv2.imshow("webcam", erodeImage)

    i += 1
    cv2.waitKey(1)

