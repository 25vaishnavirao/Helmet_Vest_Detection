
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import contextlib


MODEL_PATH = 'runs/detect/train3/weights/best.pt'  # path to your trained YOLO weights
CLASS_NAMES = ['vest', 'helmet']                  
COLORS = {                                          # BGR color for each class when drawing boxes
    'vest': (0, 255, 255),      # Yellow (BGR format)
    'helmet': (255, 0, 0)       # Blue
}
IOU_THRESHOLD = 0.9         
CONFIDENCE_THRESHOLD = 0.9   


def suppress_yolo_logs(model_path=MODEL_PATH):
    """Load the YOLO model while suppressing stdout logs.
    """
    # Using contextlib.redirect_stdout to temporarily silence prints
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                model = YOLO(model_path)
    except Exception as e:
        # If loading fails, surface useful error for debugging
        print(f"Error loading model from {model_path}: {e}")
        raise
    return model


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two axis-aligned boxes.

    box format: [x1, y1, x2, y2] (pixel coordinates)

    IoU measures overlap between boxes. We use it to decide if two
    detections are actually the same object.
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # compute intersection rectangle
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # compute areas of individual boxes
    box1_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    box2_area = max(0, (x2_p - x1_p)) * max(0, (y2_p - y1_p))

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(boxes, scores, threshold=IOU_THRESHOLD):
    """Simple NMS implementation.

    - boxes: list of [x1, y1, x2, y2]
    - scores: list of confidence scores
    - threshold: IoU cutoff to suppress overlapping boxes

    Returns a list of indices to KEEP.

    NMS keeps the most confident box and removes overlapping weaker ones.
    This avoids duplicate boxes on the same object.
    """
    if not boxes:
        return []

    # Convert to numpy for convenience
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)

    # Order by score (descending)
    order = scores_np.argsort()[::-1].tolist()
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        suppressed = []
        for j in order:
            iou = compute_iou(boxes_np[i], boxes_np[j])
            if iou >= threshold:
                suppressed.append(j)

        # Remove suppressed indices from order
        order = [idx for idx in order if idx not in suppressed]

    return keep


def process_frame(frame, model):
    """Run model inference on a frame, filter low-confidence anchors, and apply NMS.

    Returns: (boxes, confidences, class_ids, keep_indices)
      - boxes: list of [x1,y1,x2,y2]
      - confidences: list of floats
      - class_ids: list of ints
      - keep_indices: list of indices (into boxes/confidences/class_ids) to keep after NMS

    For each frame, we run YOLO, filter by confidence, then apply IoU+NMS.
    The result is a set of final detections to draw. 
    """
    # Run detection 
    results = model(frame, verbose=False)

    # ultralytics returns a sequence of results (one per image). We have a single image.
    if len(results) == 0:
        return [], [], [], []
    res = results[0]

    boxes = []
    confidences = []
    class_ids = []

    # "res.boxes" contains per-box information; iterate through them
    for box in res.boxes:
        # box.conf and box.cls may be tensors; convert to float/int
        conf = float(box.conf) if hasattr(box, 'conf') else float(box.conf[0])
        cls = int(box.cls) if hasattr(box, 'cls') else int(box.cls[0])

        # Confidence filtering
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # xyxy may be accessible as box.xyxy; it's often a tensor with shape (1,4)
        xyxy = box.xyxy[0] if hasattr(box, 'xyxy') else box.xyxy
        x1, y1, x2, y2 = map(int, map(float, xyxy))

        boxes.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(cls)

    # Apply custom NMS (works across all classes; for class-aware NMS, run per-class)
    keep = non_max_suppression(boxes, confidences, IOU_THRESHOLD)

    return boxes, confidences, class_ids, keep


def draw_detections(frame, boxes, confidences, class_ids, keep):
    """Draw bounding boxes and labels on the frame for indexes in `keep`.

    Here we draw the final boxes and the confidence score so viewers can
    see what the model predicts in real-time.
    """
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        class_id = class_ids[i]
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        color = COLORS.get(label, (0, 255, 0))  # default green if not defined

        # Draw rectangle (box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare and draw text label with confidence
        text = f"{label} {confidences[i]:.2f}"
        # Put text above the box if there's space, otherwise put it inside
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# ------------------------- MAIN PIPELINE -------------------------

def main():
    """Main function: load model, open video source, and run the real-time loop.

    PRESENTATION (high-level script):
      - "We load the trained model, then open a video stream from a webcam or IP camera."
      - "Each frame is processed: model predicts boxes, we filter by confidence, apply IoU+NMS,
         draw final boxes, and display the result until the user quits."
    """
    model = suppress_yolo_logs(MODEL_PATH)
    

    # Ask user for camera source. "0" means local webcam.
    source = input("Enter IP camera URL or '0' for webcam: ").strip()
    source = int(source) if source == '0' else source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source. Check your webcam or IP stream URL.")
        return

    print("Video source opened successfully. Press 'q' to quit.")

    # Optional: measure FPS for demo
    fps_display_interval = 1  # seconds
    frame_count = 0
    start_time = cv2.getTickCount()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame. Ending stream.")
            break

        # Process the frame (model inference + filtering + NMS)
        boxes, confidences, class_ids, keep = process_frame(frame, model)

        # Draw final detections on the frame
        frame = draw_detections(frame, boxes, confidences, class_ids, keep)

        # (Optional) compute and show FPS on the frame for demo purposes
        frame_count += 1
        if frame_count % 10 == 0:
            # compute elapsed time in seconds
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Real-Time Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
