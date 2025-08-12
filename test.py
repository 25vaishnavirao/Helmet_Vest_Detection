from ultralytics import YOLO

# Load the trained model for vest and helmet detection
model = YOLO("runs/detect/train3/weights/best.pt")  

# Perform detection on test images
results = model.predict(
    source=r"C:\Users\tplma\Documents\TPL\Usecase\Helmet-vest-detection\test_images",  # path to your test images folder
    show=True,         # display results in a window 
    save=True,         # save results in runs/detect/predict
    conf=0.8          # confidence threshold (optional - default is 0.25)
)
