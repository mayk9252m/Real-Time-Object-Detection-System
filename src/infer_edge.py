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