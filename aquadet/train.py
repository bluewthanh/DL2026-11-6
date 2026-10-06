"""Fine-tune a COCO-pretrained YOLO checkpoint."""

import json
import time
from pathlib import Path

import yaml

from aquadet import ROOT
from aquadet.config import resolve

WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"


def pretrained_weights(config: dict) -> Path:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return WEIGHTS_DIR / config["model"]["weights"]


def train(config: dict, data_yaml: Path, run_name: str, device: str) -> Path:
    from ultralytics import YOLO

    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        raise ValueError(f"{run_dir} already exists; pass --name for a new run, or use `evaluate`/`benchmark --run` to re-score it")
    model = YOLO(str(pretrained_weights(config)))
    started = time.perf_counter()
    model.train(data=str(data_yaml), project=str(RUNS_DIR), name=run_name, exist_ok=False,
                device=device, verbose=True, **config["train"])
    elapsed = time.perf_counter() - started

    run_dir = Path(model.trainer.save_dir)
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8")
    info = {"wall_time_s": round(elapsed, 1),
            "epochs_completed": model.trainer.epoch + 1,
            "best_fitness": float(model.trainer.best_fitness or 0.0),
            "best_epoch": best_epoch(run_dir / "results.csv"),
            "pretrained_weights": str(pretrained_weights(config).relative_to(ROOT)),
            "data_yaml": str(resolve(data_yaml))}
    (run_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return run_dir


def best_epoch(results_csv: Path) -> int | None:
    """Epoch with the highest Ultralytics fitness (0.1*mAP50 + 0.9*mAP50-95 on val)."""
    import csv

    if not results_csv.is_file():
        return None
    with results_csv.open(newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        return None
    def fitness(row: dict) -> float:
        return 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])
    return int(float(max(rows, key=fitness)["epoch"]))
