# vest-helmet-detection
PS C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
PS C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection> cd "C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection"
PS C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection> .\venv\Scripts\Activate
(venv) PS C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection> pip install utralytics 


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Validating C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection\runs\detect\train3\weights\best.pt...
Ultralytics 8.3.170  Python-3.10.11 torch-2.5.1+cu121 CUDA:0 (NVIDIA GeForce GTX 1650 with Max-Q Design, 4096MiB)
Model summary (fused): 72 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:41<00:00,  
                   all       1427       4792      0.927      0.908      0.963      0.776
                  vest       1163       2094      0.908      0.903      0.956      0.776
                helmet       1149       2698      0.946      0.914       0.97      0.776
Speed: 1.0ms preprocess, 19.3ms inference, 0.0ms loss, 2.8ms postprocess per image
Results saved to C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection\runs\detect\train3

import os
import cv2
import numpy as np
import contextlib
from ultralytics import YOLO
import streamlit as st

# ---------------- CONFIG ----------------
MODEL_PATH = 'runs/detect/train3/weights/best.pt'
CLASS_NAMES = ['vest', 'helmet']
COLORS = {
    'vest': (0, 255, 255),   # Yellow
    'helmet': (255, 0, 0)    # Blue
}
IOU_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.8

# ---------------- UTILS ----------------
def suppress_yolo_logs(model_path=MODEL_PATH):
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
    results = model(frame, verbose=False)
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

def draw_detections(frame, boxes, confidences, class_ids, keep):
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])
        color = COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidences[i]:.2f}"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ---------------- STREAMLIT APP ----------------
def run_streamlit_app():
    st.set_page_config(page_title="Helmet & Vest Detection", layout="wide")

    # Custom CSS for fresh look
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
        .stTitle {
            text-align: center;
            color: #FF4B4B;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Real-Time Helmet & Vest Detection")

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
        model = suppress_yolo_logs()
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

            boxes, confidences, class_ids, keep = process_frame(frame, model)
            frame = draw_detections(frame, boxes, confidences, class_ids, keep)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if stop_btn:
                status_text.markdown("**Status:** Detection stopped.")
                break

        cap.release()

if __name__ == "__main__":
    run_streamlit_app()
