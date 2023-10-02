import cv2
import numpy as np
import json

#Blue color range
lowerMask = np.array([90, 20, 20])
upperMask = np.array([140, 255, 255])

#Red color range (Not working correctly)
lowerCorners = np.array([0, 0, 0])
upperCorners = np.array([50, 50, 50])

#Position (Buttom left x and y, and width and height to the right and up)
x = 0
y = 0
w = 0
h = 0

#All valid positions
savedPoints = []

#Pointer to webcam
video = cv2.VideoCapture(0)

#Counter
i = 0

#"Game loop" Where the magic happens
while True:
    #Saves video as image, success = bool, img = image
    success, img = video.read()
    #Connverts colors to HSV
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Makes a mask image in black white, where white is colors in range of lower- & upperCorners
    mask = cv2.inRange(image, lowerMask, upperMask)
    #Makes mask but for red (Not working corretly)
    maskCorner = cv2.inRange(image, lowerCorners, upperCorners)
    #Sets saves point to null
    savedPoints = []

    #Saves all cornor positions or white shapes in mask in contours (corners of shape, only square) and hierarchy saves the relationships each contour have to each other
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Draws a box around the white shapes for visuals and saves data points in savedPoints
    if len(contours) != 0:
        for contour in contours:
            #Calculates the area of the shape
            if cv2.contourArea(contour) > 50:
                #Gets the position of the area/box/white shape
                x, y, w, h = cv2.boundingRect(contour)
                print("Data: x = {0}, y = {1}, w = {2}, h = {3}".format(x, y, w, h))
                #Saves the data in a JSON format
                data = {
                    "name": "Ground",
                    "data": [x, y, w, h]
                }
                #Adds it to the array of points/data
                savedPoints.append(data)
                #Draws the rectangle shape around the white shape for visuals
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    #Saves the data points to a JSON file
    with open('data.json', 'w') as outfile:
        json.dump(savedPoints, outfile)

    print(savedPoints)

    #Shows the 3 images
    cv2.imshow("mask", mask)
    #cv2.imshow("maskCorner", maskCorner)
    cv2.imshow("webcam", img)

    i += 1
    cv2.waitKey(1)

