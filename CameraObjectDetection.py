import cv2
import numpy as np
import json

lowerMask = np.array([110, 170, 20])
upperMask = np.array([125, 255, 255])

lowerCorners = np.array([0, 0, 0])
upperCorners = np.array([50, 50, 50])
x = 0
y = 0
w = 0
h = 0

savedPoints = []

video = cv2.VideoCapture(0)

i = 0

while i < 60:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lowerMask, upperMask)
    maskCorner = cv2.inRange(image, lowerCorners, upperCorners)
    savedPoints = []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                print("Data: x = {0}, y = {1}, w = {2}, h = {3}".format(x, y, w, h))
                data = {
                    "name": "Ground",
                    "data": [x, y, w, h]
                }
                savedPoints.append(data)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    with open('CameraObjectDetection/data.json', 'w') as outfile:
        json.dump(savedPoints, outfile)

    print(savedPoints)

    cv2.imshow("mask", mask)
    cv2.imshow("maskCorner", maskCorner)
    cv2.imshow("webcam", img)

    i += 1
    cv2.waitKey(1)

