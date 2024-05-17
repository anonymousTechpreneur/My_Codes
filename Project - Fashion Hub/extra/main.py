import cv2

import numpy as np
import cv2.aruco as aruco

def findArucoMarkers(img,markerSize=6,totalMarkers=250,draw=True):
    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict= aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()
    bboxs,ids,rejected=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)
    #print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        if (bboxs != []):
            int_corners = np.int0(bboxs)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            aruco_perimeter = cv2.arcLength(bboxs[0], True)
        # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 40
            print(bboxs)
            print(pixel_cm_ratio)
        else:
            print("no")
    return [bboxs,ids]
def main():
    img = r"C:\Users\kapil\Downloads\WhatsApp Image 2021-08-22 at 12.22.47 (1).jpeg"
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    findArucoMarkers(img)



if __name__== "__main__":
    main()
