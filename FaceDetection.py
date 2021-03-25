import cv2
#import imageio
#import numpy as np

faceCas = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eyeCas =  cv2.CascadeClassifier("haarcascade-eye.xml")

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCas.detectMultiScale(gray, 1.3, 5)
    recs = []
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0), 2)
        eyes = eyeCas.detectMultiScale(gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)


if __name__ == "__main__":
    show = True
    capture = cv2.VideoCapture(0)
    while True:

        
        ret,frame = capture.read()
        recs = detect(frame)

        cv2.imshow("Face Recog", frame)
        if cv2.waitKey(1) == 27 or 0xFF== ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
#cv2.waitKey(0)
#cv2.destroyAllWindows()