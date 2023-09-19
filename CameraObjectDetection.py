import cv2
import numpy as np

lowerMask = np.array([110, 170, 20])
upperMask = np.array([125, 255, 255])

lowerCorners = np.array([0, 0, 0])
upperCorners = np.array([50, 50, 50])

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lowerMask, upperMask)
    maskCorner = cv2.inRange(image, lowerCorners, upperCorners)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 15:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
    print(contours)


    cv2.imshow("mask", mask)
    cv2.imshow("maskCorner", maskCorner)
    cv2.imshow("webcam", img)

    cv2.waitKey(1)

