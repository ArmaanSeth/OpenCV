import cv2 as cv
import mediapipe as mp
import time
import math


class poseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), -1)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if angle > 180:
            angle = 360 - angle

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), -1)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), -1)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 1)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 1)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 1)
            cv.putText(img, str(int(angle)), (x2 - 15, y2), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 150), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 150), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), -1)
            cv.putText(img,str(int(length)),(cx-2,cy-2),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)


        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    print("ENTERED CODE")
    cTime = 0
    pTime = 0

    video = r'C:\\Users\\Armaan\\OneDrive\\Desktop\\Workspace\\OpenCV\\Video\\4.mp4'
    cap = cv.VideoCapture(video)

    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)

        lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Video", img)

        if cv.waitKey(20) & 0xff == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()