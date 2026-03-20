from ultralytics import YOLO

# Load trained model later by replacing the path below
model = YOLO("runs/detect/results/baseline_run/weights/best.pt")

# Run inference on an example image
results = model.predict(
    source="data/traffic_detection/images/val",
    save=True,
    conf=0.25
)

print("Inference complete.")