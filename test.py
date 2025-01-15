from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
import cv2
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Eye Detection")
        MainWindow.setGeometry(100, 100, 800, 600)

        # Example widget: a QLabel to display status
        self.status_label = QLabel(MainWindow)
        self.status_label.setGeometry(50, 50, 300, 50)
        self.status_label.setText("No detection yet")
        self.status_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")

class EyeDetection(Ui_MainWindow):
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # Set font for OpenCV text

    def process_frame(self, frame):
        # Example logic for gaze detection
        gaze_ratio = 0.5  # Replace this with actual gaze detection logic

        if gaze_ratio < 1:
            cv2.putText(frame, "LEFT", (50, 100), self.font, 2, (0, 0, 255), 3)
            self.update_widget_property()

    def update_widget_property(self):
        """Update widget property when gaze is detected."""
        self.status_label.setText("Gaze detected: LEFT")
        self.status_label.setStyleSheet("background-color: lightblue; font-size: 16px; padding: 5px;")

def main():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = EyeDetection()
    ui.setupUi(MainWindow)
    MainWindow.show()

    # Simulate video frame processing
    frame = None  # Replace with an actual frame from a video feed
    ui.process_frame(frame)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
