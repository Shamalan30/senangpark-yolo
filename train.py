from ultralytics import YOLO

if __name__ == '__main__':
    print("Loading base YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print("Starting training with GPU...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="parking_model",
        patience=10,
        device="0",
        workers=2,
        verbose=True
    )

    print("=" * 50)
    print("TRAINING COMPLETE!")
    print("Best model saved at:")
    print("runs/detect/parking_model/weights/best.pt")
    print("=" * 50)