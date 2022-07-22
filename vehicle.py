import cv2
import numpy as np
cap=cv2.VideoCapture('video.mp4')
minw=100
minh=100
countline=550
algo= cv2.bgsegm.createBackgroundSubtractorMOG()
detect=[]
offset=6
counter=0
def centre_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

while True:
    ret, frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)
    img_sub=algo.apply(blur)
    dilat=cv2.dilate(img_sub, np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    countershape, h=cv2.findContours(dilatada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1,(15,countline),(600,countline),(255,127,0),3)
    for (i,c) in enumerate(countershape):
        (x,y,w,h)=cv2.boundingRect(c)
        valcounter=(w>=minw) and (h>minh)
        if not valcounter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle:"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)
        centre=centre_handle(x,y,w,h)
        detect.append(centre)
        cv2.circle(frame1,centre,4,(0,0,255),-1)
        for (x,y) in detect:
            if y<(countline+offset) and y>(countline-offset):
                counter+=1
                cv2.line(frame1,(15,countline),(60,countline),(127,255,0),3)
                detect.remove((x,y))
                print("Vehicle Counter:"+str(counter))
    cv2.putText(frame1,"Vehicle Counter:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    #cv2.imshow('Detector1', dilatada)
    cv2.imshow('Video Original', frame1)
    if(cv2.waitKey(1)==13):
        break
cv2.destroyAllWindows()
cap.release()


