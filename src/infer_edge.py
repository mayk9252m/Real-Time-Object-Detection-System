import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from metrics import DefectLogger