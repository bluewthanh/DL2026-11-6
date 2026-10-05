"""Closed-set YOLO benchmark on RF100 Aquarium v2. Run `python -m aquadet --help`."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Isolate from machine-wide Ultralytics settings; must be set before ultralytics is imported.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
