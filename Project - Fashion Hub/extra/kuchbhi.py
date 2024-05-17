import cv2
import numpy as np
import math
import cv2.aruco as aruco

def measurements(img,gender):
    msize_chart = [["S", 40.64, ], ["M", 43.18], ["L", 44.45], ["XL", 45.72], ["XXL", 46.99]]
    fsize_chart = [["XS", 34.29], ["S", 36.83], ["M", 39.37], ["L", 41.91], ["XL", 44.45]]
    MODE = "MPI"

    if MODE == "COCO":
        protoFile = "pose_deploy_linevec.prototxt"
        weightsFile = "pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    elif MODE == "MPI":
        protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                      [14, 11], [11, 12], [12, 13]]

    frame = cv2.imread(img)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []

    for i in range(nPoints):

        probMap = output[0, i, :, :]

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            points.append((int(x), int(y)))
        else:
            points.append((0,0))
    scale = findArucoMarkers(img)
    if(gender== "Man"):
        sd = ((math.sqrt((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2)) + (math.sqrt((points[1][0] - points[5][0]) ** 2 + (points[1][1] - points[5][1]) ** 2)))
        sd = math.ceil(sd/scale)
        sd += 6
        for i in msize_chart:
            a = i[1]
            if ((a - sd) > 0) or i[0]=="XXL":
                finl = i[0]
                break
        #print("Shoulder length=",sd)
        print("Suggested Shirt size=", finl)


    else:
        sd = ((math.sqrt((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2)) + (math.sqrt((points[1][0] - points[5][0]) ** 2 + (points[1][1] - points[5][1]) ** 2)))
        sd = math.ceil(sd/scale)
        sd += 8
        for i in fsize_chart:
            a = i[1]
            if ((a - sd) > 0 or i[0]=="XL"):
                finl = i[0]
                break
        print("Suggested Top Size=", finl)

    return frame

def age_gender(img1):
    age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")
    gender_model = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

    img = cv2.imread(img1)

    haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def detect_faces(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = haar_detector.detectMultiScale(gray, 1.3, 5)
        return face

    faces = detect_faces(img)
    for x, y, w, h in faces:
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.resize(detected_face, (224, 224))  # img shape is (224, 224, 3) now
        img_blob = cv2.dnn.blobFromImage(detected_face)  # img_blob shape is (1, 3, 224, 224)

        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        output_indexes = np.array([i for i in range(0, 101)])
        apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'

    return (apparent_predictions,gender)

def tryon(img_green,design1):
    frame = cv2.imread(img_green)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.bitwise_not(mask_white)

    # converting mask_black to 3 channels
    W, L = mask_black.shape
    mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_black_3CH[:, :, 0] = mask_black
    mask_black_3CH[:, :, 1] = mask_black
    mask_black_3CH[:, :, 2] = mask_black

    dst3 = cv2.bitwise_and(mask_black_3CH, frame)

    # ///////
    W, L = mask_white.shape
    mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
    mask_white_3CH[:, :, 0] = mask_white
    mask_white_3CH[:, :, 1] = mask_white
    mask_white_3CH[:, :, 2] = mask_white

    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)

    # /////////////////

    # changing for design
    design = cv2.imread(design1)
    design = cv2.resize(design, mask_black.shape[1::-1])

    design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)

    final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)
    cv2.imshow('final_out', final_mask_black_3CH)

    cv2.waitKey()


def findArucoMarkers(img1,markerSize=6,totalMarkers=250,draw=True):
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
            pixel_cm_ratio = aruco_perimeter / 40
        else:
            print("No ARuco found")
            return
    #print("Scalability=",pixel_cm_ratio)
    return pixel_cm_ratio
def main():

    img = r"C:\Users\kapil\Downloads\WhatsApp Image 2021-08-22 at 12.22.47 (1).jpeg"
    img_green= r"C:\Users\kapil\Downloads\Screenshot_148.jpg"
    design= r"C:\Users\kapil\Downloads\Webp.net-resizeimage.png"

    agerange=[[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80]]
    age,gender= age_gender(img)
    for x in agerange:
            if(round(age) in range(x[0],x[1])):
                print("AGE RANGE-",x)
                break;
    print("Gender-",gender)
    frame = measurements(img,gender)
    tryon(img_green, design)


if __name__== "__main__":
    main()
