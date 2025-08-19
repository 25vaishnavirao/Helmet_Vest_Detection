import sys
import os
import time
import cv2
import numpy as np
import contextlib
import threading
from typing import Callable, Optional
from ultralytics import YOLO

# ----------------- Model Config -----------------
def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

MODEL_PATH = resource_path("runs/detect/train3/weights/best.pt")
PERSON_MODEL_PATH = resource_path("yolov8n.pt")

CLASS_NAMES = ["vest", "helmet"]
IOU_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.8

def suppress_yolo_logs(model_path):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            return YOLO(model_path)

# ----------------- Helper Functions -----------------
def compute_iou(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, x2p - x1p) * max(0, y2p - y1p)
    union = a1 + a2 - inter
    return inter / union if union else 0.0

def non_max_suppression(boxes, scores, threshold=IOU_THRESHOLD):
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)
    order = scores_np.argsort()[::-1].tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        suppressed = [j for j in order if compute_iou(boxes_np[i], boxes_np[j]) >= threshold]
        order = [idx for idx in order if idx not in suppressed]
    return keep

def process_frame(frame, model):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    if len(results) == 0:
        return [], [], [], []
    res = results[0]
    boxes, confidences, class_ids = [], [], []
    for box in res.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, map(float, box.xyxy[0]))
        boxes.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(cls)
    keep = non_max_suppression(boxes, confidences)
    return boxes, confidences, class_ids, keep

def detect_safety_violations(frame, person_model, hv_model):
    person_results = person_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
    persons = [list(map(int, box.xyxy[0])) for box in person_results.boxes if int(box.cls) == 0]

    hv_boxes, hv_confs, hv_cids, hv_keep = process_frame(frame, hv_model)

    for (px1, py1, px2, py2) in persons:
        has_helmet = has_vest = False
        head_area = [px1, py1, px2, py1 + int((py2 - py1)/3)]
        torso_area = [px1, py1 + int((py2 - py1)/3), px2, py2]

        for i in hv_keep:
            x1, y1, x2, y2 = hv_boxes[i]
            cls_name = CLASS_NAMES[hv_cids[i]] if 0 <= hv_cids[i] < len(CLASS_NAMES) else str(hv_cids[i])
            if cls_name == "helmet" and compute_iou(head_area, [x1, y1, x2, y2]) > 0.15:
                has_helmet = True
            if cls_name == "vest" and compute_iou(torso_area, [x1, y1, x2, y2]) > 0.15:
                has_vest = True

        color = (0, 255, 0)
        label = "SAFE"
        if not has_helmet or not has_vest:
            label = "NOT SAFE"
            color = (0, 0, 255)

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ----------------- Threads -----------------
class CaptureThread(threading.Thread):
    def __init__(self, src, is_ip=False):
        super().__init__(daemon=True)
        self.src = src
        self.is_ip = is_ip
        self.running = threading.Event()
        self.running.set()
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        if self.is_ip:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("Cannot open video source:", self.src)
            return

        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running.clear()

class ProcessingThread(threading.Thread):
    def __init__(self, capture_thread: CaptureThread, person_model, hv_model,
                 frame_callback: Optional[Callable[[np.ndarray], None]] = None,
                 resize_to=(640, 480)):
        super().__init__(daemon=True)
        self.capture_thread = capture_thread
        self.person_model = person_model
        self.hv_model = hv_model
        self.frame_callback = frame_callback
        self.resize_to = resize_to
        self.running = threading.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            frame = self.capture_thread.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if self.resize_to:
                frame = cv2.resize(frame, self.resize_to)

           
            if not self.running.is_set():
                break

            out = detect_safety_violations(frame, self.person_model, self.hv_model)

            if not self.running.is_set():
                break

            if self.frame_callback:
                self.frame_callback(out)

    def stop(self):
        self.running.clear()

# ----------------- Orchestrator -----------------
class SafetyDetector:
    def __init__(self, model_path: str = MODEL_PATH, person_model_path: str = PERSON_MODEL_PATH):
        self.model_path = model_path
        self.person_model_path = person_model_path
        self.hv_model = suppress_yolo_logs(self.model_path)
        self.person_model = suppress_yolo_logs(self.person_model_path)
        self.capture_thread: Optional[CaptureThread] = None
        self.processing_thread: Optional[ProcessingThread] = None

    def start(self, source, frame_callback: Optional[Callable[[np.ndarray], None]] = None):
        if isinstance(source, str) and source.strip() == "0":
            source = 0
        is_ip = source != 0
        self.stop()
        self.capture_thread = CaptureThread(source, is_ip=is_ip)
        self.processing_thread = ProcessingThread(self.capture_thread, self.person_model, self.hv_model, frame_callback)
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.join(timeout=0.5)   
            self.processing_thread = None
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.join(timeout=0.5)     
            self.capture_thread = None


if __name__ == "__main__":
    src = input("Enter 0 for webcam or RTSP/HTTP URL: ").strip()
    if src == "0":
        src = 0

    det = SafetyDetector()
    last = [None]

    def cb(frame):
        last[0] = frame

    det.start(src, frame_callback=cb)
    print("Press 'q' to quit.")
    try:
        while True:
            if last[0] is not None:
                cv2.imshow("Helmet & Vest Detection", last[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
    finally:
        det.stop()
        cv2.destroyAllWindows()
