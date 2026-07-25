import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from metrics import DefectLogger


class ThreadedFrameGrabber:
    """Continuously reads frames in a background thread; always exposes the latest one.

    This is the key trick for hitting real-time FPS on edge boards: cv2.VideoCapture.read()
    blocks on decode, and if you call it inline in your inference loop, your throughput is
    capped at min(decode_fps, inference_fps) instead of inference_fps alone.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # Ask the camera driver for the smallest possible buffer so we don't
        # accumulate latency -- we always want the newest frame, not a queue.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                self.frame = frame
            if not ok:
                break

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class FPSMeter:
    """Rolling-window FPS counter, more stable than a single-frame delta."""

    def __init__(self, window=60):
        self.timestamps = deque(maxlen=window)

    def tick(self):
        self.timestamps.append(time.perf_counter())

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / span if span > 0 else 0.0

def draw_detections(frame, result, class_names, defect_conf_threshold):
    """Draw boxes; defects above the reject threshold are flagged for the line PLC/HMI."""
    flagged = False
    if result.boxes is None:
        return frame, flagged

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        is_reject = conf >= defect_conf_threshold
        flagged = flagged or is_reject
        color = (0, 0, 255) if is_reject else (0, 200, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls_id, cls_id)} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, flagged


def parse_args():
    p = argparse.ArgumentParser(description="Real-time edge inference for defect detection")
    p.add_argument("--weights", type=str, required=True,
                    help="Path to exported model: .engine (TensorRT), .onnx, .xml (OpenVINO), or .pt")
    p.add_argument("--source", type=str, default="0", help="Camera index, RTSP URL, or video file")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--reject-conf", type=float, default=0.5,
                    help="Confidence above which a detection triggers a line reject/alarm")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--frame-skip", type=int, default=0,
                    help="Run inference on every Nth frame (0 = every frame). "
                         "Use on the weakest hardware if you can't hit target FPS otherwise.")
    p.add_argument("--save", action="store_true", help="Save annotated output to runs/infer/")
    p.add_argument("--no-display", action="store_true", help="Headless mode (no cv2.imshow window)")
    p.add_argument("--log-defects", action="store_true",
                    help="Log every reject-worthy detection to logs/defect_log.csv for escape-rate tracking")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights, task="detect")
    class_names = model.names

    grabber = ThreadedFrameGrabber(source)
    fps_meter = FPSMeter()
    logger = DefectLogger(Path("logs/defect_log.csv")) if args.log_defects else None

    writer = None
    frame_idx = 0

    print("Starting inference loop. Press 'q' to quit (if display enabled).")
    try:
        while True:
            ok, frame = grabber.read()
            if not ok or frame is None:
                time.sleep(0.005)  # camera hasn't produced a frame yet, don't busy-spin
                continue

            frame_idx += 1
            if args.frame_skip and frame_idx % (args.frame_skip + 1) != 0:
                continue

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                verbose=False,
            )
            result = results[0]

            annotated, flagged = draw_detections(frame, result, class_names, args.reject_conf)
            fps_meter.tick()

            if flagged and logger:
                logger.log(result, class_names, frame_idx)

            cv2.putText(annotated, f"FPS: {fps_meter.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            if flagged:
                cv2.putText(annotated, "DEFECT DETECTED", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    out_dir = Path("runs/infer")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(out_dir / "output.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                    )
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Defect Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        grabber.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if logger:
            logger.close()
        print(f"Final rolling FPS: {fps_meter.fps:.1f}")


if __name__ == "__main__":
    main()