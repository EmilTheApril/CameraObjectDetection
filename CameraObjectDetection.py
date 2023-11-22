import cv2
import numpy as np
import json
import time
import math

#Pointer to webcam
video = cv2.VideoCapture(0)

hMin = 100
sMin = 110
vMin = 86
hMax = 122
sMax = 250
vMax = 198
blobList = []
finalBlobList = []
arrayBlobCount = 0

#Counter
i = 0

def FindBlobs(image, blobIndex, index, width, height):
    pixelCheck = []
    pixelCheck.append([blobList[blobIndex][index][0] + 1, blobList[blobIndex][index][1] + 1])
    pixelCheck.append([blobList[blobIndex][index][0] + 1, blobList[blobIndex][index][1] - 1])
    pixelCheck.append([blobList[blobIndex][index][0] - 1, blobList[blobIndex][index][1] + 1])
    pixelCheck.append([blobList[blobIndex][index][0] - 1, blobList[blobIndex][index][1] - 1])
    for i in range(len(pixelCheck)):
        if len(pixelCheck[i]) == 2:
            if image[pixelCheck[i][0], pixelCheck[i][1]] == 255:
                blobList[blobIndex].append([pixelCheck[i][0], pixelCheck[i][1]])
                image[pixelCheck[i][0], pixelCheck[i][1]] = 0
    if len(blobList[blobIndex]) > (index + 1):
        FindBlobs(image, blobIndex, (index + 1), width, height)
    else:
        print(f"Blob nr: {blobIndex}, Pixel Count: {len(blobList[blobIndex])}")

while i < 3:
    #Turns on camera and saves the image in img
    success, img = video.read()
    
    #Crops the image, to fit the whiteboard
    croppedImage = cv2.resize(img, (630, 415))
    
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    #Kernel used to enlargen all pixels
    kernel3x3 = np.ones((3, 3), np.uint8)
    
    #Enlargens all pixels
    erodeImage = cv2.dilate(mask, kernel3x3)
    
    #BloblobList holds all pixel pos
    width, height = erodeImage.shape
    print(erodeImage.shape)
    if i == 2:
        for y in range(height - 1):
            for x in range(width - 1):
                if erodeImage[x, y] == 255:
                    blobList.append([])
                    erodeImage[x, y] = 0
                    blobList[arrayBlobCount].append([x, y])
                    FindBlobs(erodeImage, arrayBlobCount, 0, width, height)
                    arrayBlobCount += 1
    
    #For final product
    #for z in range(len(blobList)):
    #    if len(blobList[z]) >= 70:
    #        finalBlobList.append(blobList[z])
    #        for i in range(len(blobList[z])):
    #            erodeImage[blobList[z][i][0], blobList[z][i][1]] = 255

    #For testing: Makes all blobs white again
    for z in range(len(blobList)):
        for i in range(len(blobList[z])):
                erodeImage[blobList[z][i][0], blobList[z][i][1]] = 255
        if len(blobList[z]) >= 70:
            finalBlobList.append(blobList[z])

    #For testing: Saves the image of all blobs before remove small ones
    cv2.imwrite("Result.png", erodeImage)

    #For testing: Makes all small blobs black
    for z in range(len(blobList)):
        if len(blobList[z]) < 70:
            for i in range(len(blobList[z])):
                erodeImage[blobList[z][i][0], blobList[z][i][1]] = 0  

    #Shows the image/video
    cv2.imshow("webcam", erodeImage)

    i += 1
    cv2.waitKey(1)

#Print the final blobs pixel sizes and how many there are
print("Final results:")
for i in range(len(finalBlobList)):
    print(f"Blob nr: {i}, Pixel Count: {len(finalBlobList[i])}")

#For testing: Saves the image of all blobs that got colored white again and are the final blobs
cv2.imwrite("final.png", erodeImage)

#Function that returns json file ready data
def BlobResultToJSONReady(pixelArray):
    name = "name"
    pos = [0, 0]
    scale = [0, 0]
    rotation = [0, 0, 0]

def find_corners_pivot_rotation_and_scale(points):
    # Calculate the pivot point (center of mass)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    pivot_point = [center_x, center_y]
    holder = pivot_point[0]
    pivot_point = [pivot_point[1], holder]

    # Find the topmost right and topmost left points
    top_right_point = max(points, key=lambda x: (x[1], -x[0]))
    top_left_point = min(points, key=lambda x: (x[1], x[0]))

    # Swap the x and y coordinates for top_right and top_left to handle the case where y-coordinates are the same
    holder = top_right_point[1]
    top_right_point = [top_left_point[1], top_right_point[0]]
    top_left_point = [holder, top_left_point[0]]

    overPivot = [(top_left_point[0] + top_right_point[0]) / 2, (top_left_point[1] + top_right_point[1]) / 2]

    #width
    width = math.sqrt((top_left_point[0] - top_right_point[0])**2 + (top_left_point[1] - top_right_point[1])**2) / 10 / 2

    #height
    height = math.sqrt((overPivot[0] - pivot_point[0])**2 + (overPivot[1] - pivot_point[1])**2) * 2 / 10 / 2

    scale = [width, height]

    #angle
    angle = math.atan2(top_left_point[1] - top_right_point[1], top_left_point[0] - top_right_point[0])
    # Convert angle from radians to degrees
    angle_degrees = math.degrees(angle)

    rotation = [0, 0, angle_degrees]

    pos = [((pivot_point[0]) - (erodeImage.shape[0])) / 50, ((erodeImage.shape[1] / 2) - pivot_point[1]) / 50]

    return pos, scale, rotation



print(find_corners_pivot_rotation_and_scale(finalBlobList[0]))

#Save blobs as json file
