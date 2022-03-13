import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='images'
images=[]

classNames=[]
myList=os.listdir(path)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("İsimler : ",classNames)

def findEcnodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # face_recognition.face_encodings resimdeki tüm yüzleri getirir. Burada ilkini almak istiyoruz --> [0]
        encodeList.append(encode)
    return encodeList

"""
def markAttedance(name):
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

"""

encodeListKnown=findEcnodings(images)
print("Encoding Complete.")

#webcam
cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurrentFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1, y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #text=str(int((1-faceDis)*100))
            benzerlik=int((1-(np.min(faceDis)))*100)
            text="% "+str(benzerlik)
            cv2.putText(img,name+text,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1)
            #markAttedance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


    cv2.imshow('WebCam', img)
    cv2.waitKey(1)

