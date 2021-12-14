import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/Video 3.mp4")
pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=5)

while True:
    cv2.namedWindow("Video Processing", cv2.WINDOW_NORMAL)
    success, vid = cap.read()
    vidRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(vidRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(vid, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = vid.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(vid, str(int(fps)), (70,550), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 10)
    cv2.imshow("Video Processing", vid)
    cv2.waitKey(1)