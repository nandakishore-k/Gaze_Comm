import dlib
print(dlib.__version__)


import cv2
import numpy as np
import dlib
from math import hypot

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    right_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    left_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom_top = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (255, 255 , 0), 2)
    ver_line = cv2.line(frame, center_top, bottom_top, (255, 255 , 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot((center_top[0] - bottom_top[0]), (center_top[1] - bottom_top[1]))

    ratio = (hor_line_length/ver_line_length)

    return ratio
    


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("H:\Main Project\shape_predictor_68_face_facial_landmarks.dat")
cap = cv2.VideoCapture(0)

font =  cv2.FONT_HERSHEY_COMPLEX


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left() , face.top()
        #x1, y1 = face.right() , face.bottom()
        #cv2.rectangle(frame,(x, y),(x1, y1), (0 , 255, 0), 2)

        landmarks = predictor(gray, face)
        
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)

        if left_eye_ratio > 6:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255,255,0))

        #y = facial_landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (255, 255 , 0), 2)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()