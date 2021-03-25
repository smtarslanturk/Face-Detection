# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:53:08 2020

@author: samet.arslanturk
"""

import cv2 
import numpy as np 

yuzCas = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")

kamera = cv2.VideoCapture(0)

while (1):
    ret, frame = kamera.read()
    griton = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    yuzler = yuzCas.detectMultiScale(griton,1.3,4)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow("Face Recog", frame)
    if cv2.waitKey(1) == 27:    #ASCII coduna gore ESC ile kapatmayı saglar. 
       break

kamera.release()    #VideoCaptura yaptığımız atamaya gore kamerayı kapatır. 
cv2.destroyAllWindows() #Kamera goruntusu için açılan frami kapatır. 


