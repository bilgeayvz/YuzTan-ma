import cv2
import mediapipe as mp
import time
#08,11,12,13
cap=cv2.VideoCapture("video/12.mp4")
pTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    success,img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACE_CONNECTIONS,
                                  drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih,iw,ic=img.shape
                x,y,z=int(lm.x*iw),int(lm.y*ih),int(lm.z*ic)
                print(id,x,y,z)

    cTime=time.time()
    fps=1/(cTime-pTime)
    cv2.putText(img, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN, 3 , (0,255,0),3)
    pTime=cTime
    cv2.imshow("Image",img)
    cv2.waitKey(10)