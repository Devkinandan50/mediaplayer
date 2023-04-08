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


# class VideoCamera(object):
	
# 	def get_frame(self):
		# global cap1
		# global df1
		# cap1 = WebcamVideoStream(src=0).start()
		# image = cap1.read()
		# image=cv2.resize(image,(600,500))
		# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		# face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		# df1 = pd.read_csv(music_dist[show_text[0]])
		# df1 = df1[['Name','Album','Artist']]
		# df1 = df1.head(15)
		# for (x,y,w,h) in face_rects:
		# 	cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
		# 	roi_gray_frame = gray[y:y + h, x:x + w]
		# 	cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
		# 	prediction = emotion_model.predict(cropped_img)

		# 	maxindex = int(np.argmax(prediction))
		# 	show_text[0] = maxindex 
		# 	#print("===========================================",music_dist[show_text[0]],"===========================================")
		# 	#print(df1)
		# 	cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		# 	df1 = music_rec()
			
		# global last_frame1
		# last_frame1 = image.copy()
		# pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		# img = Image.fromarray(last_frame1)
		# img = np.array(img)
		# ret, jpeg = cv2.imencode('.jpg', img)
		# return jpeg.tobytes(), df1

		


 

 
# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     try:
#         objs = DeepFace.analyze(imgRGB, actions = ['emotion'])
#         print(objs[0]['dominant_emotion'])
#         cv2.putText(imgRGB, str(objs[0]['dominant_emotion']), (250,150), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 12)
#         # print(objs['dominant_emotion'])
#     except:
#         print("NO FACE")


#     if hands:
#         hand = hands[0]
#         print(hand['type'])

#         if(hand['type'] == 'Right'):
#             x, y, w, h = hand['bbox']
 
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
     
#             imgCropShape = imgCrop.shape
     
#             aspectRatio = h / w
     
#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)
     
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
     
#             cv2.rectangle(imgOutput, (x - offset, y - offset-50),
#                           (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
#             cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#             cv2.rectangle(imgOutput, (x-offset, y-offset),
#                           (x + w+offset, y + h+offset), (255, 0, 255), 4)
     
#             # cv2.imshow("ImageCrop", imgCrop)
#             # cv2.imshow("ImageWhite", imgWhite)
 
#     cv2.imshow("Image", imgOutput)
#     cv2.waitKey(9)