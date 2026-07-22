import csv
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

class DefectLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._file = open(self.path, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "frame_idx", "class_name", "confidence",
                                   "x1", "y1", "x2", "y2"])  # Write header if file is new

    def log(self, result, class_names, frame_idx):
        if result.boxes is None:
            return
        ts = datetime.now().isoformat()