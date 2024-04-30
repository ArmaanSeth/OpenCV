import cv2 as cv
import mediapipe as mp
import time

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1,color=(0,255,0))

ctime=0
ptime=0

video=r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\5.mp4'
cap=cv.VideoCapture(video)

def resizeFrame(frame,scale=0.75):
    height=int(frame.shape[0]*scale)
    width=int(frame.shape[1]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

while(True):
    success,img=cap.read()
    img=resizeFrame(img,0.25)
    
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            
            for id,lm in enumerate(facelms.landmark):
                print(lm)
                h,w,c=img.shape
                x,y=int(lm.x*w),int(lm.y*h)
                print(id,x,y)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv.putText(img,str(int(fps)),(50,50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xff==ord("q"):
        break
cap.release()
cv.destroyAllWindows()