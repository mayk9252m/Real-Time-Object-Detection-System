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
        ts = datetime.now().isoformat(timespec="milliseconds")
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            self._writer.writerow([ts, frame_idx, class_names.get(cls_id, cls_id),
                                   round(conf, 4), x1, y1, x2, y2])
        self._file.flush()

    def close(self):
        self._file.close()


@dataclass
class EscapeRateResult:
    total_ground_truth_defects: int
    caught_by_model: int
    escaped: int
    escape_rate: float  # fraction, e.g. 0.078 == 7.8%


class EscapeRateCalculator:
    """
    Compares a ground-truth defect manifest against the model's detection log
    to compute defect escape rate over some batch/shift/day.

    ground_truth_ids: iterable of unique part/unit IDs known (via QA audit,
            downstream inspection, or customer return) to actually be defective.
        detected_ids: iterable of unique part/unit IDs the model flagged as defective.
    
        Both should key off the same unit identifier (e.g. serial number, barcode,
        or timestamp+station bucket) -- wiring that join is specific to your MES,
        so this is intentionally left as plain ID sets rather than parsing a fixed format.
    """

    @staticmethod
    def compute(ground_truth_ids, detected_ids) -> EscapeRateResult:
        gt = set(ground_truth_ids)
        detected = set(detected_ids)
    
        caught = gt & detected
        escaped = gt - detected
    
        total = len(gt)
        n_caught = len(caught)
        n_escaped = len(escaped)
        rate = (n_escaped / total) if total > 0 else 0.0
    
        return EscapeRateResult(
            total_ground_truth_defects=total,
            caught_by_model=n_caught,
            escaped=n_escaped,
            escape_rate=rate,
            )