import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from deepface import DeepFace

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
 
offset = 20
imgSize = 300

counter = 0
 
labels = ["Pause", "Resume", "Forward", "Backward"]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        objs = DeepFace.analyze(imgRGB, actions = ['emotion'])
        print(objs[0]['dominant_emotion'])
        cv2.putText(imgRGB, str(objs[0]['dominant_emotion']), (250,150), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 12)
        # print(objs['dominant_emotion'])
    except:
        print("NO FACE")


    if hands:
        hand = hands[0]
        print(hand['type'])

        if(hand['type'] == 'Right'):
            x, y, w, h = hand['bbox']
 
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
     
            imgCropShape = imgCrop.shape
     
            aspectRatio = h / w
     
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
     
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
     
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)
     
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
 
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(9)