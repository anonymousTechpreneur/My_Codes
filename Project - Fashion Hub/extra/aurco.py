import cv2
import numpy as np
import math
import cv2.aruco as aruco

def findArucoMarkers(img1,markerSize=4,totalMarkers=250,draw=True):
    img = cv2.imread(img1,cv2.IMREAD_COLOR)
    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict= aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()
    bboxs,ids,rejected=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        if (bboxs != []):
            int_corners = np.int0(bboxs)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            aruco_perimeter = cv2.arcLength(bboxs[0], True)
            pixel_cm_ratio = aruco_perimeter / 60
        else:
            print("No ARuco found")
            return
    #print("Scalability=",pixel_cm_ratio)
    return pixel_cm_ratio

def main():

    img = r"E:\Tanmay\main4.jpg"
    frame = findArucoMarkers(img)
    print(frame)

if __name__== "__main__":
    main()
