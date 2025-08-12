import os
import cv2
import numpy as np
import contextlib
from ultralytics import YOLO
import streamlit as st
import time


MODEL_PATH = 'runs/detect/train3/weights/best.pt'  # Your helmet/vest model
CLASS_NAMES = ['vest', 'helmet']
COLORS = {
    'vest': (0, 255, 255),   # Yellow
    'helmet': (255, 0, 0)    # Blue
}
IOU_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.8 

def suppress_yolo_logs(model_path):
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise
    return model


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    box2_area = max(0, (x2_p - x1_p)) * max(0, (y2_p - y1_p))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0.0


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
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = map(int, map(float, xyxy))
        boxes.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(cls)
    keep = non_max_suppression(boxes, confidences)
    return boxes, confidences, class_ids, keep


def detect_safety_violations(frame, person_model, hv_model):
    person_results = person_model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
    persons = [list(map(int, box.xyxy[0])) for box in person_results.boxes if int(box.cls) == 0]

    hv_boxes, hv_confidences, hv_class_ids, hv_keep = process_frame(frame, hv_model)

    violations = []

    for (px1, py1, px2, py2) in persons:
        has_helmet = False
        has_vest = False

        for i in hv_keep:
            x1, y1, x2, y2 = hv_boxes[i]
            cls_name = CLASS_NAMES[hv_class_ids[i]]

            head_area = [px1, py1, px2, py1 + int((py2 - py1) / 3)]
            if cls_name == 'helmet' and compute_iou(head_area, [x1, y1, x2, y2]) > 0.15:
                has_helmet = True

            torso_area = [px1, py1 + int((py2 - py1) / 3), px2, py2]
            if cls_name == 'vest' and compute_iou(torso_area, [x1, y1, x2, y2]) > 0.15:
                has_vest = True

        color = (0, 255, 0)
        label = "SAFE"
        if not has_helmet or not has_vest:
            label = "NOT SAFE"
            color = (0, 0, 255)
            violations.append((px1, py1, px2, py2))

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, violations


def run_streamlit_app():
    st.set_page_config(page_title="Helmet & Vest Safety Detection", layout="wide")

    st.markdown("""
        <style>
        .stButton button {
            background-color: #0073e6;
            color: white;
            font-size: 16px;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Real-Time Helmet & Vest Safety Detection")

    col1, col2 = st.columns([1, 2])

    with col1:
        source_type = st.radio("Select Video Source", ["Webcam", "IP Camera"])
        ip_url = None
        if source_type == "IP Camera":
            ip_url = st.text_input("Enter IP Camera URL")

        start_btn = st.button("Start Detection")
        stop_btn = st.button("Stop Detection")

    with col2:
        stframe = st.empty()
        status_text = st.empty()

    if start_btn:
        status_text.markdown("**Status:** Running detection...")
        hv_model = suppress_yolo_logs(MODEL_PATH)
        person_model = suppress_yolo_logs('yolov8n.pt')
        source = 0 if source_type == "Webcam" else ip_url
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            st.error("Could not open video source.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            frame, violations = detect_safety_violations(frame, person_model, hv_model)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Violation warning removed here

            if stop_btn:
                status_text.markdown("**Status:** Detection stopped.")
                break

        cap.release()


if __name__ == "__main__":
    run_streamlit_app()
