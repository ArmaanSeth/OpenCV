import cv2 as cv
import mediapipe as mp
import time

cTime=0
pTime=0


mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

video=r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\4.mp4'
cap=cv.VideoCapture(video)

while True:
    success,img=cap.read()
    # img=cv.resize(img,(int(img.shape[1]*0.25),int(img.shape[0]*0.25)))

    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results=pose.process(imgRGB)

    # print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        print(results.pose_landmarks)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id,lm)

            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(img,(cx,cy),10,(255,0,255))
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime


    cv.putText(img,str(int(fps)),(50,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv.imshow("Video",img)

    if cv.waitKey(20) & 0xff==ord('q'):
        break


cap.release()
cv.destroyAllWindows()
