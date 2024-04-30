import numpy as np
import time
import cv2 as cv
import poseDetectionModule as pm

def resizeFrame(frame,scale=0.75):
    height=int(frame.shape[0]*scale)
    width=int(frame.shape[1]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

ctime=0
ptime=0

path='C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\AiTrainer\\test\\1.mp4'
cap=cv.VideoCapture(0)

countL=0.0
dirL=0
colorL=(0,255,0)
countR=0.0
dirR=0
colorR=(0,255,0)
detector=pm.poseDetector()

while(True):
    success,img=cap.read()
    img=cv.flip(img,1)
    # img=resizeFrame(img,0.25)

    h,w,c=img.shape

    img=detector.findPose(img,False)
    lmList=detector.findPosition(img,False)


    if len(lmList)!=0:
        #Left arm
        angleL=detector.findAngle(img,11,13,15)
        #Right arm
        angleR=detector.findAngle(img,12,14,16)
        perL=100-int(np.interp(angleL,(25,160),(0,100)))
        perR=100-int(np.interp(angleR,(25,160),(0,100)))
        
        
        if perR==100:
            if dirR==0:
                countR+=0.5
                dirR=-1
                colorR=(255,0,200)      
        if perR==0:
            if dirR==-1: 
                countR+=0.5
                dirR=0
                colorR=(0,255,0)
        
        if perL==100:
            if dirL==0:
                countL+=0.5
                dirL=-1
                colorL=(255,0,200)
        if perL==0:
            if dirL==-1: 
                countL+=0.5
                dirL=0
                colorL=(0,255,0)

        # print(count)
        cv.putText(img,str(countR),(10,95),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv.rectangle(img,(20,300-2*perR),(40,300),colorR,-1)
        cv.rectangle(img,(20,100),(40,300),(0,0,0),2)

        cv.putText(img,str(countL),(w-70,95),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv.rectangle(img,(w-35,300-2*perL),(w-55,300),colorL,-1)
        cv.rectangle(img,(w-35,100),(w-55,300),(0,0,0),2)


    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,f'FPS:{int(fps)}',(500,50),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
