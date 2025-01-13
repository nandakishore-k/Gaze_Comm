import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV and PyQt Integration")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display video frames
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # OpenCV video capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam

        # Timer to update frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms (~33 fps)
    #--------------functions-------------------------------------------------
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
   

    def update_frame(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        cap = cv2.VideoCapture(0)

        font =  cv2.FONT_HERSHEY_COMPLEX

        #ret, frame = self.cap.read()
#--------------------------------------------------------------------------------------
        ret, frame = self.cap.read()
        new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            #x, y = face.left() , face.top()
            #x1, y1 = face.right() , face.bottom()
            #cv2.rectangle(frame,(x, y),(x1, y1), (0 , 255, 0), 2)

            landmarks = predictor(gray, face)
            
            #blinking detection ------------------------------------------------------------------------------
            left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
            right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)

            blinking_ratio = (left_eye_ratio + right_eye_ratio)/2

            if blinking_ratio > 5.7:
                cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255,255,0))

            #gaze detection-----------------------------------------------------------------------------------
            
            gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)

            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

            

            if(gaze_ratio < 1):
                cv2.putText(frame, "LEFT", (50,100), font, 2, (0,0, 255), 3)
                new_frame[:] = (0, 0, 255)
            elif 1 < gaze_ratio < 3:
                cv2.putText(frame, "CENTER", (50,100), font, 2, (255,0, 255), 3)
                new_frame[:] = (255, 0, 255)
            else:
                cv2.putText(frame, "RIGHT", (50,100), font, 2, (0,255, 255), 3)
                new_frame[:] = (0, 255, 255)


            
            
            cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0,255, 255), 3)













        if ret:
            # Convert OpenCV BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Create QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap and display in QLabel
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Cleanup resources on close."""
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
