# ui.py
import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout

from detection import SafetyDetector 

class HelmetVestUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helmet & Vest Detection")
        self.setGeometry(80, 80, 960, 680)
        self.setStyleSheet("""
            QWidget {background-color: #ffffff; color: #212121; font-family: Arial;}
            QPushButton {background-color: #2196f3; border-radius: 12px; padding: 10px 20px; font-size: 16px; color: white;}
            QPushButton:hover {background-color: #1976d2;}
            QLabel {font-size: 16px; color: #212121;}
            QLineEdit {padding: 10px; font-size: 16px; border: 1px solid #bbb; border-radius: 10px; background-color: #f0f0f0; color: #212121;}
        """)

        title = QLabel("Helmet & Vest Detection ⛑️🦺")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976d2; margin: 10px 0;")

        self.video_label = QLabel("Video Feed")
        self.video_label.setFixedSize(900, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #e0e0e0; border-radius: 15px; border: 2px solid #bbb;")

        self.input_src = QLineEdit()
        self.input_src.setPlaceholderText("Enter 0 for webcam or RTSP/HTTP URL for IP camera")

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.status = QLabel("Status: Idle")
        self.status.setFont(QFont("Arial", 14))
        self.status.setStyleSheet("color: #ff5722;")

        ctl = QHBoxLayout()
        ctl.addWidget(self.input_src)
        ctl.addWidget(self.start_btn)
        ctl.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(ctl)
        layout.addWidget(self.video_label)
        layout.addWidget(self.status)
        self.setLayout(layout)

        # core detector
        self.detector = SafetyDetector()
        self.last_frame = None

        # wire buttons
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # timer to paint frames delivered by the callback (no queue)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.paint_last_frame)
        self.timer.start(33)  # ~30 FPS refresh

    # callback used by core to supply frames
    def on_frame(self, frame):
        self.last_frame = frame

    def paint_last_frame(self):
        if self.last_frame is None:
            return
        frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def start_detection(self):
        src_text = self.input_src.text().strip() or "0"
        self.detector.start(src_text, frame_callback=self.on_frame)
        self.status.setText("Status: Running")
        self.video_label.setText("")  # clear placeholder

    def stop_detection(self):
        self.detector.stop()
        self.last_frame = None
        self.video_label.clear()
        self.video_label.setText("Video Feed")
        self.status.setText("Status: Stopped")

    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    w = HelmetVestUI()
    w.show()
    sys.exit(app.exec_())
