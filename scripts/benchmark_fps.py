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


def main():
    args = parse_args()
    model = YOLO(args.weights, task="detect")

    # Synthetic frame -- fine for pure throughput measurement since we're timing
    # the forward pass + NMS, not decode quality.
    dummy = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
    batch = [dummy] * args.batch

    print(f"Warming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        model.predict(batch, imgsz=args.imgsz, device=args.device, half=args.half, verbose=False)

    print(f"Benchmarking ({args.iters} iters, batch={args.batch})...")
    latencies = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        model.predict(batch, imgsz=args.imgsz, device=args.device, half=args.half, verbose=False)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)
    mean_latency_ms = latencies.mean() * 1000
    p99_latency_ms = np.percentile(latencies, 99) * 1000
    throughput_fps = args.batch / latencies.mean()
