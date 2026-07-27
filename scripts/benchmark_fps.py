import argparse
import time

import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args()