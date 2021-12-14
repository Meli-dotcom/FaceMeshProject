import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=5)

    def findFace(self, vid, draw=True):
        self.vidRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.vidRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(vid, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = vid.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # cv2.putText(vid, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #                      1, (255, 0, 0), 1)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)

        return vid, faces


def main():
    cap = cv2.VideoCapture("Videos/Video 3.mp4")
    pTime = 0
    cTime = 0
    detector = faceDetector(maxFaces = 1)
    while True:
        cv2.namedWindow("Video Processing", cv2.WINDOW_NORMAL)
        success, vid = cap.read()
        vid, faces = detector.findFace(vid, False)
        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(vid, str(int(fps)), (70, 550), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 10)
        cv2.imshow("Video Processing", vid)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()