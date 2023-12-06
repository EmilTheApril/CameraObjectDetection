import cv2
import numpy as np
import json
import time
import math
import sys
import asyncio
import os.path

sys.setrecursionlimit(10000)

#Function that returns json file ready data
def BlobResultToJSONReady(pixelArray):
    name = "name"
    pos = [0, 0]
    scale = [0, 0]
    rotation = [0, 0, 0]

def FindBlobs(image, blobList, blobIndex, index, width, height):
    pixelCheck = []
    pixelCheck.append([blobList[blobIndex][index][0] + 1, blobList[blobIndex][index][1]])
    pixelCheck.append([blobList[blobIndex][index][0] - 1, blobList[blobIndex][index][1]])
    pixelCheck.append([blobList[blobIndex][index][0], blobList[blobIndex][index][1] + 1])
    pixelCheck.append([blobList[blobIndex][index][0], blobList[blobIndex][index][1] - 1])
    for i in range(len(pixelCheck)):
        if (len(pixelCheck[i]) == 2 and pixelCheck[i][0] > 0 and pixelCheck[i][0] < width and pixelCheck[i][1] > 0 and pixelCheck[i][1] < height):
            if image[pixelCheck[i][0], pixelCheck[i][1]] == 255:
                blobList[blobIndex].append([pixelCheck[i][0], pixelCheck[i][1]])
                image[pixelCheck[i][0], pixelCheck[i][1]] = 0
    if len(blobList[blobIndex]) > (index + 1):
        return FindBlobs(image, blobList, blobIndex, (index + 1), width, height)
    else:
        print(f"Blob nr: {blobIndex}, Pixel Count: {len(blobList[blobIndex])}")
        return blobList

def find_corners_pivot_rotation_and_scale(points, image, tlCorner, widthMultiplier, heightMultiplier):
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
    width = math.sqrt((top_left_point[0] - top_right_point[0])**2 + (top_left_point[1] - top_right_point[1])**2)

    #height
    height = math.sqrt((overPivot[0] - pivot_point[0])**2 + (overPivot[1] - pivot_point[1])**2)

    scale = [width * widthMultiplier, height * heightMultiplier]

    #angle
    angle = math.atan2(top_left_point[1] - top_right_point[1], top_left_point[0] - top_right_point[0])
    # Convert angle from radians to degrees
    angle_degrees = math.degrees(angle)

    rotation = [0, 0, angle_degrees]

    pos = [(pivot_point[0] - tlCorner[0]) * heightMultiplier, (pivot_point[1] - tlCorner[1]) * widthMultiplier]

    name = "Ground"

    return name, pos, scale, rotation

def GetCornerPivot(points):
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    pivot_point = [center_x, center_y]
    holder = pivot_point[0]
    pivot_point = [pivot_point[1], holder]
    return pivot_point

def CornerPointsScreenScaleConverter(cornerPoints):
    rect = np.zeros((4, 2), dtype = "float32")

    for i in range(len(cornerPoints)):
        rect[i] = GetCornerPivot(cornerPoints[i])

    s = rect.sum(axis = 1)
    diff = np.diff(rect, axis = 1)

    tl = rect[np.argmin(s)]
    tr = rect[np.argmin(diff)]
    bl = rect[np.argmax(diff)]
    br = rect[np.argmax(s)]

    widthMultiplier = 1920 / ((math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2) + math.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)) / 2)
    heightMultiplier  = 1080 / ((math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2) + math.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)) / 2)

    return widthMultiplier, heightMultiplier, tl
    

def TakePictureAndApplyEffects(img, colorHSV):
    #Crops the image, to fit the whiteboard
    croppedImage = cv2.resize(img, (630, 415))

    hMin = colorHSV[0]
    hMax = colorHSV[1]
    sMin = colorHSV[2]
    sMax = colorHSV[3]
    vMin = colorHSV[4]
    vMax = colorHSV[5]
    
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    #Kernel used to enlargen all pixels
    kernel3x3 = np.ones((3, 3), np.uint8)
    
    #Enlargens all pixels
    erodeImage = cv2.dilate(mask, kernel3x3)

    return erodeImage

def GrassFireAlgorithm(image, blobList, arrayBlobCount):
    width, height = image.shape

    for y in range(height - 1):
            for x in range(width - 1):
                if image[x, y] == 255:
                    blobList.append([])
                    image[x, y] = 0
                    blobList[arrayBlobCount].append([x, y])
                    blobList = FindBlobs(image, blobList, arrayBlobCount, 0, width, height)
                    arrayBlobCount += 1
    return blobList, image, arrayBlobCount

def AddBLOBsToFinalBLOBList(blobList, image, threshold, finalBlobList):
    for z in range(len(blobList)):
        if len(blobList[z]) >= threshold:
            finalBlobList.append(blobList[z])
            for i in range(len(blobList[z])):
                image[blobList[z][i][0], blobList[z][i][1]] = 255
    return finalBlobList, image

def AddBLOBsToFinalBLOBListTesting(blobList, image, threshold, finalBlobList, imageOutputName):
    for z in range(len(blobList)):
        for i in range(len(blobList[z])):
                image[blobList[z][i][0], blobList[z][i][1]] = 255
        if len(blobList[z]) >= threshold:
            finalBlobList.append(blobList[z])
    
    imageWithAllBLOBs = image.copy()

    for z in range(len(blobList)):
        if len(blobList[z]) < threshold:
            for i in range(len(blobList[z])):
                image[blobList[z][i][0], blobList[z][i][1]] = 0
    
    cv2.imwrite(imageOutputName, imageWithAllBLOBs)
    cv2.imshow(imageOutputName, imageWithAllBLOBs)

    return finalBlobList, image

def PrintFinalResults(finalBlobList):
    print("Final results:")
    for i in range(len(finalBlobList)):
        print(f"Blob nr: {i}, Pixel Count: {len(finalBlobList[i])}")

def RunPython():
    #Pointer to webcam
    video = cv2.VideoCapture(0)

    blueColor = [100, 110, 110, 255, 100, 255]
    redColor = [158, 179, 80, 210, 120, 229]

    blobList = []
    blobListCorners = []
    finalBlobList = []
    finalBlobListCorners = []
    arrayBlobCount = 0
    arrayBlobCountCorners = 0
    image = None
    imageCorners = None
    widthMultiplier = 0
    heightMultiplier = 0
    tlCorner = []

    #Counter
    i = 0

    while i < 4:
        #Turns on camera and saves the image in img
        success, img = video.read()

        if i == 3:
            image = TakePictureAndApplyEffects(img, blueColor)
            imageCorners = TakePictureAndApplyEffects(img, redColor)
            blobList, image, arrayBlobCount = GrassFireAlgorithm(image, blobList, arrayBlobCount)
            blobListCorners, imageCorners, arrayBlobCountCorners = GrassFireAlgorithm(imageCorners, blobListCorners, arrayBlobCountCorners)

            #For final product
            #finalBlobList, image = AddBLOBsToFinalBLOBList(image, 70, finalBlobList)

            #For testing
            finalBlobList, image = AddBLOBsToFinalBLOBListTesting(blobList, image, 70, finalBlobList, "ResultWithAllBlobs.png")
            finalBlobListCorners, imageCorners = AddBLOBsToFinalBLOBListTesting(blobListCorners, imageCorners, 250, finalBlobListCorners, "ResultWithAllBlobsCornor.png")

            widthMultiplier, heightMultiplier, tlCorner = CornerPointsScreenScaleConverter(finalBlobListCorners)

        i += 1
    
    #Print the final blobs pixel sizes and how many there are
    PrintFinalResults(finalBlobList)

    outputList = []
    output_path = os.getcwd() + "/Assets/data.json"

    if(len(finalBlobList) > 0):
        for i in range(len(finalBlobList)):
            name, pos, scale, rotation = find_corners_pivot_rotation_and_scale(finalBlobList[i], image, tlCorner, widthMultiplier, heightMultiplier)
            jsonData = {
                "name": name,
                "pos": pos,
                "scale": scale,
                "rotation": rotation 
            }
            outputList.append(jsonData);
    else:
        print("No BLOBs found")

    with open(output_path, "w") as outfile:
        json.dump(outputList, outfile)

async def CheckIfFileExists():
    path = os.getcwd() + "/Assets/start.txt"
    if(os.path.isfile(path)):
        os.remove(path)
        return True
    else:
        return False

async def main():
    fileFound = False
    while fileFound == False:
        print("Program running")

        fileFound = await CheckIfFileExists()

        await asyncio.sleep(1)
    RunPython()

asyncio.run(main())