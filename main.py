import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

sound=pyglet.media.load("sound.wav",streaming=False)
left_sound=pyglet.media.load("left.wav",streaming=False)
right_sound=pyglet.media.load("right.wav",streaming=False)
def draw_menu():
    rows, cols, _ = frame.shape
    # Draw the selection rectangle for the menu
    cv2.rectangle(frame, (0, 0), (cols // 2, rows), (255, 255, 255), -1)
    cv2.rectangle(frame, (cols // 2, 0), (cols, rows), (200, 200, 200), -1)

    # Add text for "Left" option
    cv2.putText(frame, "LEFT", (int(cols * 0.1), int(rows * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    # Add text for "Right" option
    cv2.putText(frame, "RIGHT", (int(cols * 0.6), int(rows * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
def eyes_contour_points(landmarks):
    # Extract points for left and right eyes based on dlib's 68-point model
    left_eye = np.array([
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(37).x, landmarks.part(37).y),
        (landmarks.part(38).x, landmarks.part(38).y),
        (landmarks.part(39).x, landmarks.part(39).y),
        (landmarks.part(40).x, landmarks.part(40).y),
        (landmarks.part(41).x, landmarks.part(41).y)
    ], np.int32)

    right_eye = np.array([
        (landmarks.part(42).x, landmarks.part(42).y),
        (landmarks.part(43).x, landmarks.part(43).y),
        (landmarks.part(44).x, landmarks.part(44).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(46).x, landmarks.part(46).y),
        (landmarks.part(47).x, landmarks.part(47).y)
    ], np.int32)

    return left_eye, right_eye

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    right_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    left_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom_top = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (255, 255 , 0), 2)
    #ver_line = cv2.line(frame, center_top, bottom_top, (255, 255 , 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot((center_top[0] - bottom_top[0]), (center_top[1] - bottom_top[1]))

    ratio = (hor_line_length/ver_line_length)

    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)])
        

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2 )
    cv2.fillPoly(mask, [eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask= mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])


    gray_eye = eye[min_y: max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold =  threshold_eye[0: height, 0: int(width/2)]
    right_side_threshold =  threshold_eye[0: height,int(width/2): width]


    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    left_side_threshold = cv2.resize(left_side_threshold, None,fx =5, fy=5)
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = cv2.resize(right_side_threshold, None,fx =5, fy=5)
    right_side_white = cv2.countNonZero(right_side_threshold)

    cv2.polylines(frame, [eye_region], True, (0, 0, 255), 2 )

    
    gaze_ratio = left_side_white/(right_side_white + 0.000000001)

    return gaze_ratio
    


board=np.zeros((500,500),np.uint8)
board[:]=255
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/Users/nsys/Desktop/main project/Gaze_Comm/shape_predictor_68_face_landmarks.dat")
keyboard=np.zeros((600,1000,3),np.uint8)
#keyboard = np.zeros((300, 500, 3), np.uint8)
keys_set_1={0:"Q",1:"W",2:"E",3:"R",4:"T",5:"A",6:"S",7:"D",8:"F",9:"G"
,10:"Z",11:"X",12:"C",13:"V",14:"<"}
keys_set_2={0:"Y",1:"U",2:"I",3:"O",4:"P",5:"H",6:"J",7:"K",8:"L",9:"_"
,10:"V",11:"B",12:"N",13:"M",14:"<"}
def letter(letter_index,text,letter_light):
    #keys
    rows = letter_index // 5  # To decide row based on index
    cols = letter_index % 5   # To decide column based on index
    x = cols * 200  # Cell width is 200
    y = rows * 200  # Cell height is 200
    if letter_index==0:
        x=0
        y=0
    elif letter_index==1:
        x=200
        y=0
    elif letter_index==2:
        x=400
        y=0
    elif letter_index==3:
        x=600
        y=0
    elif letter_index==4:
        x=800
        y=0
    elif letter_index==5:
        x=0
        y=200
    elif letter_index==6:
        x=200
        y=200
    elif letter_index==7:
        x=400
        y=200
    elif letter_index==8:
        x=600
        y=200
    elif letter_index==9:
        x=800
        y=200
    elif letter_index==10:
        x=0
        y=400
    elif letter_index==11:
        x=200
        y=400
    elif letter_index==12:
        x=400
        y=400
    elif letter_index==13:
        x=600
        y=400
    elif letter_index==14:
        x=800
        y=400
    
    width=200
    height=200
    th=3
    if letter_light is True:
        cv2.rectangle(keyboard,(x+th,y+th),(x+width-th,y+height-th),(255,255,255),-1)
    else:
        cv2.rectangle(keyboard,(x+th,y+th),(x+width-th,y+height-th),(255,0,0),th)
    font_letter=cv2.FONT_HERSHEY_PLAIN
    font_scale=10
    font_th=4
    text_size=cv2.getTextSize(text,font_letter,font_scale,font_th)
    width_text,height_text=text_size[0]
    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y
    cv2.putText(keyboard,text,(text_x,text_y),font_letter,font_scale,(255,0,0),font_th)

#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

font =  cv2.FONT_HERSHEY_COMPLEX

frames=0
letter_index=0
blinking_frames=0
frames_to_blink=6
frames_active_letter=9
text=""
keyboard_selected="left"
last_keyboard_selected="left"
select_keyboard_menu=True
keyboard_selection_frames=0
while True:
    _, frame = cap.read()
    #frame=cv2.resize(frame,None,fx=0.5,fy=0.5)
    rows,cols,_=frame.shape
    keyboard[:]=(26,26,26)
    frames+=1
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame[rows-50:rows,0:cols]=(255,255,255)
    if select_keyboard_menu is True:
        draw_menu()
    if keyboard_selected=="left":
        keys_set=keys_set_1
    else:
        keys_set=keys_set_2
    active_letter=keys_set[letter_index]
    faces = detector(gray)
    for face in faces:
        #x, y = face.left() , face.top()
        #x1, y1 = face.right() , face.bottom()
        #cv2.rectangle(frame,(x, y),(x1, y1), (0 , 255, 0), 2)

        landmarks = predictor(gray, face)
        left_eye,right_eye=eyes_contour_points(landmarks)
        #blinking detection ------------------------------------------------------------------------------
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)

        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2
        cv2.polylines(frame,[left_eye],True,(0,0,255),2)
        cv2.polylines(frame,[right_eye],True,(0,0,255),2)

        """if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255,255,0),thickness=3)
            blinking_frames+=1
            frames-=1
            if blinking_frames==5:
                text+=active_letter
                sound.play()
                time.sleep(1)

        else:
            blinking_frames=0"""
          
        #gaze detection-----------------------------------------------------------------------------------
        if select_keyboard_menu is True:
            gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)

            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        

            if(gaze_ratio < 1):
            #cv2.putText(frame, "LEFT", (50,100), font, 2, (0,0, 255), 3)
            #new_frame[:] = (0, 0, 255)
                keyboard_selected="right"
                keyboard_selection_frames+=1
                if keyboard_selection_frames==15:
                    select_keyboard_menu=False
                    right_sound.play()
                    frames=0
                    #keyboard_selection_frames=0
                if keyboard_selected!=last_keyboard_selected:
                    #right_sound.play()
                    #time.sleep(1)
                    last_keyboard_selected=keyboard_selected
                    keyboard_selection_frames=0
            else:
                #cv2.putText(frame, "RIGHT", (50,100), font, 2, (0,255, 255), 3)
                #new_frame[:] = (0, 255, 255)
                keyboard_selected="left"
                keyboard_selection_frames+=1
                if keyboard_selection_frames==15:
                    select_keyboard_menu=False
                    left_sound.play()
                    frames=0
                if keyboard_selected!=last_keyboard_selected:
                    #left_sound.play()
                    #time.sleep(1)
                    last_keyboard_selected=keyboard_selected
                    keyboard_selection_frames=0
        else:
            print(f"Blinking Ratio: {blinking_ratio}")
            if blinking_ratio>5.7:
                cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255,255,0),thickness=3)
                print(active_letter)
                blinking_frames+=1
                frames-=1
                cv2.polylines(frame,[left_eye],True,(0,0,255),2)
                cv2.polylines(frame,[right_eye],True,(0,0,255),2)
                if blinking_frames==frames_to_blink:
                    if active_letter!="<" and active_letter!="_":
                        text+=active_letter
                        
                    if active_letter=="_":
                        text+=" "
                    sound.play()
                    select_keyboard_menu=True
            else:
                blinking_frames=0
        if select_keyboard_menu is False:
            if frames==frames_active_letter:
                letter_index+=1
                frames=0
            if letter_index==15:
                letter_index=0
            for i in range(15):
                if i==letter_index:
                    light=True
                else:
                    light=False
                letter(i,keys_set[i],light)
        
        cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0,255, 255), 3)

        #cv2.imshow("Treshold", threshold_eye)
        #cv2.imshow("left_eye", left_eye)
        #cv2.imshow("left_threshold", left_side_threshold)
        #cv2.imshow("right_threshold", right_side_threshold)
        ##eye = cv2.resize(gray_eye, None,fx =5, fy=5)
        #cv2.imshow("Eye",eye)

        #y = facial_landmarks.part(36).y
        #cv2.circle(frame, (x, y), 3, (255, 255 , 0), 2)

        #show direction---------------------------------------------------------------------------------------
        
    if frames==15:
        letter_index+=1
        frames=0
    if letter_index==15:
        letter_index=0
    for i in range(15):
        if i==letter_index:
            light=True
        else:
            light=False
        letter(i,keys_set[i],light)
    cv2.putText(board,text,(10,100),font,4,0,3)
    percentage_blinking=blinking_frames/frames_to_blink
    loading_x=int(cols*percentage_blinking)
    cv2.rectangle(frames,(0,rows-50),(loading_x,rows),(51,51,51),-1)

    #cv2.imshow("newFr", new_frame)
    cv2.imshow("Frame",frame)
    cv2.imshow("Virtual keyboard",keyboard)
    cv2.imshow("Board",board)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()