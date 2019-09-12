import numpy as np
import cv2

# C:\\Dev\\Python\\Plain\\face-identifier-detector\\testopencv\src\\cascades\\data\\haarcascade_frontalface_alt2.xml

face_cascade = cv2.CascadeClassifier(
    'C:\\Dev\\Python\\Plain\\face-identifier-detector\\testopencv\src\\cascades\\data\\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        img_item = 'img.png'
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + y
        end_cord_y = x + y
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
