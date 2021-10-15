#Import the libs which we need
import cv2
import time
import mediapipe as mp 

cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://entrada:Luisfelipe10@192.168.1.101/1')
#rtsp://username:password@192.168.1.64/1
ret, img = cap.read()
print(ret)

#Display the img
cv2.imshow("HandTracking",img)
cv2.waitKey(1)