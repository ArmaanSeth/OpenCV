import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self,static_image_mode=False,max_num_faces=2,refine_landmarks=False,min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode
        self.max_num_faces=max_num_faces
        self.refine_landmarks=refine_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,self.refine_landmarks,
                                               self.min_detection_confidence,self.min_tracking_confidence)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1,color=(0,255,0))

    
    def findFaceMesh(self,img,draw=True):

        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        results=self.faceMesh.process(imgRGB)
        faces=[]
        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,facelms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                face=[]
                for id,lm in enumerate(facelms.landmark):
                    # print(lm)
                    h,w,c=img.shape
                    x,y=int(lm.x*w),int(lm.y*h)
                    face.append([x,y])
                    cv.putText(img,str(id),(x,y),cv.FONT_HERSHEY_PLAIN,0.7,(0,255,0),1)
                    # print(id,x,y)
                faces.append(face)
        return img,faces

def resizeFrame(frame,scale=0.75):
        height=int(frame.shape[0]*scale)
        width=int(frame.shape[1]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def main():        
    ctime=0
    ptime=0

    video=r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\7.mp4'
    cap=cv.VideoCapture(video)  
    detector=FaceMeshDetector()

    while(True):
        success,img=cap.read()
        
        img=resizeFrame(img,0.25)
        img,faces=detector.findFaceMesh(img)
        
        if len(faces)!=0:
            print(len(faces))

        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv.putText(img,str(int(fps)),(50,50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        cv.imshow("Video",img)
        if cv.waitKey(1) & 0xff==ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__=="__main__":
    main()