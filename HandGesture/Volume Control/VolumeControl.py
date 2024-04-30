import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
wCam,hCam=640,480
#############################



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)

minVol=volRange[0]
maxVol=volRange[1]


ctime=0
ptime=0
cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector=htm.HandDetector(detectionCon=0.7)

def getvolume(v,maxl,minl):    
    # Hand Range - 25-185
    # Vol Range - -65.5-0

    range1=maxVol-minVol
    range2=maxl-minl
    ratio=range1/range2
    return (v-minl)*ratio+minVol

while(True):
    success,img=cap.read()

    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    # print(lmList)
    if len(lmList)!=0:
        # print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=int((x1+x2)/2),int((y1+y2)/2)

        cv.circle(img,(x1,y1),3,(255,0,0),2)
        cv.circle(img,(x2,y2),3,(255,0,0),2)
        cv.line(img,(x1,y1),(x2,y2),(0,255,255),2)
        cv.circle(img,(cx,cy),5,(255,0,0),2)

        maxl=190
        minl=25

        length=math.hypot(x2-x1,y2-y1)
        # print(length)
        vol=getvolume(length,maxl,minl)
        if(length<minl):
            cv.circle(img,(cx,cy),5,(0,255,0),-1)
            vol=minVol
        elif length>maxl:
            cv.circle(img,(cx,cy),5,(0,0,255),-1)
            vol=maxVol

        # print(int(getvolume(length,maxl,minl)))


        volume.SetMasterVolumeLevel(vol,None)

        cv.putText(img,f'{int(100*(vol-minVol)/(maxVol-minVol))}%',(50,145),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv.rectangle(img,(50,350-int(200*(vol-minVol)/(maxVol-minVol))),(85,350),(255,0,0),-1)        
        cv.rectangle(img,(50,150),(85,350),(0,0,0),3)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,f'FPS:{int(fps)}',(50,50),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()