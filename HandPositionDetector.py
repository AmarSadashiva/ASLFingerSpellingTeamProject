import cv2
import mediapipe as mp
import numpy as np

# To detect hands in the video and to return the coordinates to crop the hand.


class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    # To obtain and draw the hand landmarks
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # Returns coordinates to crop the hand
    def findPosition(self, img, handNo = 0, draw = True):
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            cxList = []
            cyList = []
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cxList.append(cx)
                cyList.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            return min(cxList), max(cxList), min(cyList),max(cyList)
        else:
            return w/2,w/2,h/2,h/2

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    xMinList = []
    xMaxList = []
    yMinList = []
    yMaxList = []
    success, img = cap.read()
    while success:
        success, img = cap.read()
        if success:
            img = cv2.flip(img,1)
            img = detector.findHands(img)
            xMin, xMax, yMin, yMax = detector.findPosition(img)
            xMinList.append(xMin)
            xMaxList.append(xMax)
            yMinList.append(yMin)
            yMaxList.append(yMax)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
    print("minx = ", min(xMinList))
    print("maxx = ", max(xMaxList))
    print("miny = ", min(yMinList))
    print("maxy = ", max(yMaxList))
    print()
    
    


if __name__ == "__main__":
    main()

