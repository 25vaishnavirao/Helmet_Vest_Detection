import os

BASE_DIR = r"C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection\dataset"
SUBFOLDERS = ['train', 'valid', 'test']
VEST_CLASS_ID = '0'
HELMET_CLASS_ID = '1'
VEST_IOU_THRESHOLD = 0.8
HELMET_IOU_THRESHOLD = 0.3

def compute_iou(box1, box2):
    def to_corners(box):
        x, y, w, h = box
        return x - w/2, y - h/2, x + w/2, y + h/2

    x1_min, y1_min, x1_max, y1_max = to_corners(box1)
    x2_min, y2_min, x2_max, y2_max = to_corners(box2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(inter_x_max - inter_x_min, 0)
    inter_h = max(inter_y_max - inter_y_min, 0)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

# whole image in YOLO coords: center 0.5,0.5 w=1,h=1
whole_image_box = [0.5, 0.5, 1.0, 1.0]

for subfolder in SUBFOLDERS:
    print(f"\nCleaning dataset: {subfolder}")
    LABELS_DIR = os.path.join(BASE_DIR, subfolder, 'labels')
    OUTPUT_DIR = LABELS_DIR + "_cleaned"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_vest, removed_vest = 0, 0
    total_helmet, removed_helmet = 0, 0

    for file_name in os.listdir(LABELS_DIR):
        if not file_name.endswith(".txt"):
            continue
        input_path = os.path.join(LABELS_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        with open(input_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x, y, w, h = parts
            box = [float(x), float(y), float(w), float(h)]

            if class_id == VEST_CLASS_ID:
                iou = compute_iou(box, whole_image_box)
                if iou > VEST_IOU_THRESHOLD:
                    print(f"Removed large vest box in {file_name} (IOU={iou:.2f})")
                    removed_vest += 1
                    continue
                else:
                    total_vest += 1
            elif class_id == HELMET_CLASS_ID:
                iou = compute_iou(box, whole_image_box)
                if iou > HELMET_IOU_THRESHOLD:
                    print(f"Removed large helmet box in {file_name} (IOU={iou:.2f})")
                    removed_helmet += 1
                    continue
                else:
                    total_helmet += 1
            new_lines.append(line)

        with open(output_path, "w") as f:
            for l in new_lines:
                f.write(l if l.endswith('\n') else l+'\n')

    print(f"Done cleaning {subfolder}:")
    print(f" Vest boxes kept: {total_vest}, removed: {removed_vest}")
    print(f" Helmet boxes kept: {total_helmet}, removed: {removed_helmet}")
    print(f" Cleaned labels saved to: {OUTPUT_DIR}")
