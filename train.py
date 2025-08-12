from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection\dataset\data.yaml",
        epochs=50,
        imgsz=640,
        device="cuda"
    )
