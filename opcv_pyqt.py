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

    def update_frame(self):
        ret, frame = self.cap.read()
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
