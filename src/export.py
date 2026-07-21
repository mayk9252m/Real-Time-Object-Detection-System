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