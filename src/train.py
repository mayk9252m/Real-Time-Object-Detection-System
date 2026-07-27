import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLOv8 for defect detection")
    p.add_argument("--data", type=str, default="config/dataset.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Base checkpoint: yolov8n/s/m/l/x.pt, or a .yaml to train from scratch")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640,
                    help="Lower (e.g. 480/416) trades accuracy for edge FPS headroom")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="'0' for GPU 0, 'cpu' for CPU)")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    p.add_argument("--project", type=str, default="runs/defect_detect")
    p.add_argument("--name", type=str, default="yolov8_defect")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset config not found at {args.data}. "
            "Populate data/images and data/labels first (see config/dataset.yaml)."
        )

    model = YOLO(args.model)

    # Augmentation choices matter a lot for defect detection: defects are often
    # small, low-contrast, and class-imbalanced (most parts are non-defective).
    # mosaic/mixup help with imbalance; conservative color jitter avoids washing
    # out subtle discoloration/contamination signals.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.01,     # small hue jitter -- avoid corrupting discoloration defect signal
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,    # parts on a line are roughly upright; don't overdo rotation aug
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,     # usually parts have a consistent orientation on the line
        close_mosaic=10,  # disable mosaic for the last N epochs to stabilize convergence
        val=True,
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("\nTraining complete.")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    
    # Quick validation summary on the held-out val split
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    
    
if __name__ == "__main__":
    main()