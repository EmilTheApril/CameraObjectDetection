import cv2
import numpy as np
import json
from PIL import Image

#Pointer to webcam
video = cv2.VideoCapture(0)

#Counter
i = 0

while i < 3:
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
    
    #Bloblist holds all pixel pos
    blobList = [[]]
    arrayBlobCount = 0
    count = 0
    print(erodeImage.shape)
    width, height = 670, 1030
    
    if i == 2:
        for y in range(height):
            for x in range(width):
                print(erodeImage[x, y])
                if erodeImage[x, y] == 255:
                    print("Yes")
                    blobList[arrayBlobCount].append([x, y])
                    
                    count = 0
                    while count < len(blobList[arrayBlobCount]):
                        pixelCheck = []
                        if blobList[arrayBlobCount][count][0] + 1 <= width & blobList[arrayBlobCount][count][1] + 1 <= height:
                            pixelCheck.append(erodeImage[blobList[arrayBlobCount][count][0] + 1, blobList[arrayBlobCount][count][1] + 1])
                        if blobList[arrayBlobCount][count][0] + 1 <= width & blobList[arrayBlobCount][count][1] + 1 >= 0:
                            pixelCheck.append(erodeImage[blobList[arrayBlobCount][count][0] + 1, blobList[arrayBlobCount][count][1] - 1])
                        if blobList[arrayBlobCount][count][0] + 1 >= 0 & blobList[arrayBlobCount][count][1] + 1 <= height:
                            pixelCheck.append(erodeImage[blobList[arrayBlobCount][count][0] - 1, blobList[arrayBlobCount][count][1] + 1])
                        if blobList[arrayBlobCount][count][0] + 1 >= 0 & blobList[arrayBlobCount][count][1] + 1 >= 0:
                            pixelCheck.append(erodeImage[blobList[arrayBlobCount][count][0] - 1, blobList[arrayBlobCount][count][1] - 1])
                        for i in range(len(pixelCheck)):
                            if pixelCheck[i] == 255:
                                blobList[arrayBlobCount].append(pixelCheck[i])
                        count += 1
                            
        print(blobList[0])
    
    #Shows the image/video
    cv2.imshow("webcam", erodeImage)

    i += 1
    cv2.waitKey(1)

