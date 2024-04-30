import cv2 as cv
import mediapipe as mp
import time


def resizeFrame(frame,scale=0.75):
    height=int(frame.shape[0]*scale)
    width=int(frame.shape[1]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)



mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)

cTime=0
pTime=0

video=r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\7.mp4'
cap=cv.VideoCapture(video)
while True:
    success,img=cap.read()
    img=resizeFrame(img,0.25)

    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)

    # print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # print(id,detection)
            print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img,detection)
            bboxC=detection.location_data.relative_bounding_box
            h,w,c=img.shape
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv.rectangle(img,bbox,(255,0,255),2)
            cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
     
    cv.putText(img,str(int(fps)),(50,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv.imshow("Video",img)
    
    if cv.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv.destroyAllWindows()


