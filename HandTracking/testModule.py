import cv2 as cv
import numpy as np
import time as time
import HandTrackingModule as hnd

cTime=0
pTime=0
capture=cv.VideoCapture(0)
detector=hnd.HandDetector()
while(True):
    sucess,img=capture.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    if len(lmList)!=0:
        print(lmList)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    
    cv.imshow("Image",img)
    if cv.waitKey(20) & 0xff==ord('q'):
        break
capture.release()
cv.destroyAllWindows()