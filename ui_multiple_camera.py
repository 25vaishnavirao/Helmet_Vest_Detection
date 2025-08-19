import sys
import cv2
from functools import partial
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from detection2 import SafetyDetector  


class MultiHelmetVestUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helmet & Vest Detection — Multi Camera")
        self.setGeometry(50, 50, 1600, 800)  
        self.setStyleSheet("""
            QWidget { background-color: #ffffff; color: #212121; font-family: Arial; }
            QPushButton { background-color: #2196f3; border-radius: 12px; padding: 12px 24px; font-size: 18px; color: white; }
            QPushButton:hover { background-color: #1976d2; }
            QLabel { font-size: 18px; color: #212121; }
            QLineEdit { padding: 14px; font-size: 18px; border: 1px solid #bbb; border-radius: 12px;
                        background-color: #f7f7f7; color: #212121; min-width: 300px; }
        """)

        # -------- UI Setup --------
        title_row = self.setup_title()
        inputs_row = self.setup_inputs()
        ctl_row = self.setup_controls()
        video_row = self.setup_video_feeds()

        # -------- Root Layout --------
        root = QVBoxLayout()
        root.addLayout(title_row)
        root.addSpacing(10)
        root.addLayout(inputs_row)
        root.addLayout(ctl_row)
        root.addSpacing(12)
        root.addLayout(video_row)
        self.setLayout(root)

        # Detection engines
        self.detectors = [None, None, None]
        self.last_frames = [None, None, None]

        # Signals
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)

        # Refresh timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.paint_frames)
        self.timer.start(33)  

    # ---------- Setup sections ----------
    def setup_title(self):
        """Top title and logo"""
        title_label = QLabel("Helmet & Vest Detection ⛑️🦺")
        title_label.setFont(QFont("Arial Black", 72, QFont.Bold))  
        title_label.setStyleSheet("color: #1976d2; font-size: 72px;")
        title_label.setAlignment(Qt.AlignCenter) 

        logo_label = QLabel()
        pix = QPixmap("TPL_logo.jpeg").scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pix)
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row = QHBoxLayout()
        row.addStretch()         
        row.addWidget(title_label, stretch=1, alignment=Qt.AlignCenter)
        row.addWidget(logo_label, alignment=Qt.AlignRight | Qt.AlignVCenter)
        return row

    def setup_inputs(self):
        """Camera input fields + Show/Hide button"""
        self.input_webcam = QLineEdit("0")
        self.input_webcam.setPlaceholderText("Webcam (0, 1, ...)")
        self.input_ip1 = QLineEdit()
        self.input_ip1.setPlaceholderText("IP Camera 1 (RTSP/HTTP URL)")
        self.input_ip1.setEchoMode(QLineEdit.Password)
        self.input_ip2 = QLineEdit()
        self.input_ip2.setPlaceholderText("IP Camera 2 (RTSP/HTTP URL)")
        self.input_ip2.setEchoMode(QLineEdit.Password)

        self.show_ip_btn = QPushButton("Show IP URLs")
        self.show_ip_btn.setCheckable(True)
        self.show_ip_btn.toggled.connect(self.toggle_ip_visibility)

        row = QHBoxLayout()
        row.addWidget(self.input_webcam)
        row.addWidget(self.input_ip1)
        row.addWidget(self.input_ip2)
        row.addWidget(self.show_ip_btn)
        return row

    def setup_controls(self):
        """Start/Stop buttons"""
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addStretch()
        return row

    def setup_video_feeds(self):
        """Video feed panels and status labels"""
        self.video_labels, self.status_labels = [], []
        self.cam_names = ["Webcam", "IP Cam 1", "IP Cam 2"]

        row = QHBoxLayout()
        for name in self.cam_names:
            container = QVBoxLayout()
            vlabel = QLabel(f"{name} Feed")
            vlabel.setFixedSize(640, 480) 
            vlabel.setAlignment(Qt.AlignCenter)
            vlabel.setStyleSheet("background-color: #e0e0e0; border-radius: 16px; border: 2px solid #bbb;")
            slabel = QLabel("Status: Idle")
            slabel.setFont(QFont("Arial", 16)) 
            slabel.setAlignment(Qt.AlignCenter)
            slabel.setStyleSheet("color: #ff5722;")

            container.addWidget(vlabel)
            container.addWidget(slabel)
            self.video_labels.append(vlabel)
            self.status_labels.append(slabel)
            row.addLayout(container)
        return row

    # ---------- Utility methods ----------
    def toggle_ip_visibility(self, checked):
        mode = QLineEdit.Normal if checked else QLineEdit.Password
        self.input_ip1.setEchoMode(mode)
        self.input_ip2.setEchoMode(mode)
        self.show_ip_btn.setText("Hide IP URLs" if checked else "Show IP URLs")

    def update_status(self, idx, text, color):
        """Update status label with text and color"""
        self.status_labels[idx].setText(f"Status: {text}")
        self.status_labels[idx].setStyleSheet(f"color: {color};")

    def reset_feed(self, idx):
        """Reset video feed label when stopped"""
        self.video_labels[idx].clear()
        self.video_labels[idx].setText(f"{self.cam_names[idx]} Feed")

    def create_detector(self, src, idx):
        """Start a detector for a given source"""
        try:
            det = SafetyDetector()
            det.start(src, frame_callback=partial(self.on_frame, idx))
            self.detectors[idx] = det
            self.update_status(idx, "Running", "#4caf50")
            self.video_labels[idx].setText("")
            return True
        except Exception as e:
            self.detectors[idx] = None
            self.update_status(idx, "Error", "#e53935")
            QMessageBox.critical(self, "Camera Error", f"Failed to start source {idx+1}:\n{e}")
            return False

    def on_frame(self, idx, frame):
        self.last_frames[idx] = frame

    def paint_frames(self):
        for idx, frame in enumerate(self.last_frames):
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_labels[idx].width(),
                self.video_labels[idx].height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_labels[idx].setPixmap(pix)

    def start_detection(self):
        self.stop_detection()
        sources = [self.input_webcam.text().strip(), self.input_ip1.text().strip(), self.input_ip2.text().strip()]
        any_started = False
        for idx, src in enumerate(sources):
            if not src:
                self.update_status(idx, "Idle", "#ff9800")
                continue
            if src == "0":
                src = 0
            if self.create_detector(src, idx):
                any_started = True
        if not any_started:
            QMessageBox.information(self, "No Sources", "Please enter at least one camera source.")

    def stop_detection(self):
        for idx, det in enumerate(self.detectors):
            if det is not None:
                try:
                    det.stop()
                except Exception:
                    pass
                self.detectors[idx] = None
            self.last_frames[idx] = None
            self.reset_feed(idx)
            self.update_status(idx, "Stopped", "#ff5722")

    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MultiHelmetVestUI()
    w.show()
    sys.exit(app.exec_())
