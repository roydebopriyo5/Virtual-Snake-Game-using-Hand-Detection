import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import random
import winsound

cap = cv2.VideoCapture(0)
cap.set(3, 1280)   # frame width prop id and size
cap.set(4, 720)    # frame height prop id and size

detector = HandDetector(detectionCon=0.8, maxHands=1)

class Snake_Game:
    def __init__(self, pathFood):
        self.points = []           # all points of the snake
        self.lenghts = []          # distance b/w each points
        self.currentLength = 0     # total length of the snake
        self.allowedLength = 90   # total allowed length
        self.previousHead = 0, 0   # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.heightFood, self.widthFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.foodRandomLocation()

        self.score = 0
        self.gameOver = False

    def foodRandomLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.gameOver:
            cv2.putText(imgMain, "Game Over!", (330, 280), cv2.FONT_HERSHEY_COMPLEX, 3, (139,0,139), 5)
            cv2.putText(imgMain, f'Score: {self.score}', (330, 430), cv2.FONT_HERSHEY_COMPLEX, 3, (139,0,139), 5)
            cv2.putText(imgMain, "Restart 'R' or Exit 'E' ", (50, 650), cv2.FONT_HERSHEY_COMPLEX, 1, (0,140,255), 2)

        else :
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lenghts.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # reducing length
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lenghts):
                    self.currentLength -= length
                    self.lenghts.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break
            
            # checking if snake ate food
            rx, ry = self.foodPoint
            if (rx - self.widthFood // 2) < cx < (rx + self.widthFood // 2) and (ry - self.heightFood // 2) < cy < (ry + self.heightFood // 2):
                #print('ate')
                self.foodRandomLocation()
                self.allowedLength += 50
                self.score += 1
                #print(self.score)
            
            # drawing snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i-1], self.points[i], (160,158,95), 20)
                cv2.circle(imgMain, self.points[-1], 20, (45,82,160), cv2.FILLED)
            
            # drawing food
            rx, ry = self.foodPoint
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.widthFood //2, ry - self.heightFood // 2))

            cv2.putText(imgMain, f'Score: {self.score}', (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,140,255), 2)

            # checking for collision
            point_s = np.array(self.points[:-2], np.int32)
            point_s = point_s.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [point_s], False, (160,158,95), 3)
            minDist = cv2.pointPolygonTest(point_s, (cx,cy), True)
            #print(minDist)

            if -1 <= minDist <= 1:
                #print('self hit')
                winsound.Beep(500, 500)
                self.gameOver = True
                self.points = []           # all points of the snake
                self.lenghts = []          # distance b/w each points
                self.currentLength = 0     # total length of the snake
                self.allowedLength = 150   # total allowed length
                self.previousHead = 0, 0   # previous head point
                self.foodRandomLocation()

        return imgMain

game = Snake_Game("food.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
        

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)
    if key == ord('r') or key == ord('R'):
        game.gameOver = False
        game.score = 0
    elif key == ord('e') or key == ord('E'):
        break
