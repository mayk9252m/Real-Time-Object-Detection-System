import argparse
from pathlib import Path

from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 model for edge inference")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--format", type=str, default="onnx",
                    choices=["onnx", "engine", "openvino", "ncnn", "torchscript"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 export -- ~2x speedup on most edge GPUs")
    p.add_argument("--int8", action="store_true",
                    help="INT8 quantization (engine/openvino) -- needs a calibration set, biggest speedup")
    p.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (disable for max speed)")
    p.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    p.add_argument("--workspace", type=int, default=4, help="TensorRT builder workspace, GB")
    p.add_argument("--batch", type=int, default=1, help="Fixed batch size baked into the export")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)

    kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        batch=args.batch,
    )
    if args.format == "engine":
        kwargs["workspace"] = args.workspace
        kwargs["int8"] = args.int8
    if args.format == "openvino":
        kwargs["int8"] = args.int8

    exported_path = model.export(**kwargs)

    print(f"\nExported model: {exported_path}")
    print("Sanity-check it with scripts/benchmark_fps.py before wiring it into the "
          "production inference pipeline -- export flags (half/int8/imgsz) trade "
          "accuracy for speed and should be validated on a held-out set.")

    if __name__ == "__main__":
        main()