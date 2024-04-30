import cv2 as cv
import mediapipe as mp
import time


class faceDetector:
    def __init__(self,min_detection_confidence=0.5, model_selection=0):

        self.min_detection_confidence=min_detection_confidence
        self.model_selection=model_selection
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection(self.min_detection_confidence,self.model_selection)

    def findFaces(self,img,draw=True):

        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)

        # print(results)
        bboxs=[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                # print(id,detection)
                print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                # mpDraw.draw_detection(img,detection)
                bboxC=detection.location_data.relative_bounding_box
                h,w,c=img.shape
                if draw:
                    bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                    bboxs.append([id,bbox,detection.score])
                    img=self.fancyDraw(img,bbox)
                    cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        return img,bboxs

    def fancyDraw(self,img,bbox,t=10):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        l=int(w/3)

        cv.rectangle(img,bbox,(255,0,255),2)
        cv.line(img,(x,y),(x+l,y),(255,0,254),t)
        cv.line(img,(x,y),(x,y+l),(255,0,254),t)
        cv.line(img,(x1,y1),(x1-l,y1),(255,0,254),t)
        cv.line(img,(x1,y1),(x1,y1-l),(255,0,254),t)

        return img

def resizeFrame(frame,scale=0.75):
    height=int(frame.shape[0]*scale)
    width=int(frame.shape[1]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def main():

    cTime=0
    pTime=0
    
    video=r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\7.mp4'
    cap=cv.VideoCapture(video)
    
    while True:
        success,img=cap.read()
        img=resizeFrame(img,0.25)

        face=faceDetector()
        img,bboxs=face.findFaces(img)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime    
        cv.putText(img,str(int(fps)),(50,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv.imshow("Video",img)
        
        if cv.waitKey(1) & 0xff==ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()