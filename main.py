import cv2
import mediapipe as mp
from deepface import DeepFace

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4,2)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # emotion detection
    try:
        objs = DeepFace.analyze(img, actions = ['emotion'])
        print(objs[0]['dominant_emotion'])
        # print(objs['dominant_emotion'])
    except:
        print("NO FACE")

    cv2.putText(img, str(objs[0]['dominant_emotion']), (250,150), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 12)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    lefthand = results.multi_handedness

    # print(lefthand.label)
    # get x, y point in handspoint array in pixel format
    if(lefthand != None):
        if(lefthand[0].classification[0].label == 'Left'):
            print(lefthand[0].classification[0].label)
            if multiLandMarks:
                handPoints = []
                for handLms in multiLandMarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
                    for idx, lm in enumerate(handLms.landmark):
                        # print(idx,lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        handPoints.append((cx, cy))
            
            # draw circle on handPoints with (0, 0, 255) red colour
                for point in handPoints:
                    cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)
        
                upCount = 0
                for coordinate in fingerCoordinates:
                    if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                        upCount += 1
                if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0]:
                    upCount += 1
        
                cv2.putText(img, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0), 12)
                

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(1)


