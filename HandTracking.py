#Import the libs which we need
import cv2
import time
import mediapipe as mp 

'''
@This is a simple class implementation of hand tracking
It has the modules findHands and findPositions
'''
class handTracking:
    def __init__(self, mode = False, maxHands=2, dC = 0.5, tC=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = dC
        self.trackConf = tC

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionConf,
                                        min_tracking_confidence=self.trackConf)
                                        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)

        if(self.results.multi_hand_landmarks):
            #Walk through the number of hands 
            for handLms in self.results.multi_hand_landmarks:
                if(draw):
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if(self.results.multi_hand_landmarks):
            hand = self.results.multi_hand_landmarks[handNo]

            #Walk through all the landmarks and print the ID as well the X,Y,Z
            for id, lm in enumerate(hand.landmark):
                h,w,c = img.shape
                #Change domain from float numbers to pixels
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])   
                if(draw):
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return lmList 


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    
    tracker = handTracking()

    while(True):
        _, img = cap.read()
        img = tracker.findHands(img)

        landMarks = tracker.findPosition(img)
        if(len(landMarks)!=0):
            print(landMarks)

        #Calculate the FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #Draw FPS 
        cv2.putText(img, 'FPS: '+str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (124,252,0), 3 )

        #Display the img
        cv2.imshow("HandTracking",img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

    
    