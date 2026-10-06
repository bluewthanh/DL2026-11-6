"""Model size and batch-1 latency on pre-loaded images. Only compare latencies from the same GPU."""

import json
import statistics
import time
from pathlib import Path

import numpy as np


def model_size(weights: Path, imgsz: int) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(weights))
    model.fuse(verbose=False)
    return {"params_M": round(get_num_params(model.model) / 1e6, 3),
            "gflops": round(get_flops(model.model, imgsz), 2),
            "weights_MB": round(weights.stat().st_size / 2**20, 2)}


def load_images(image_dir: Path, limit: int) -> list[np.ndarray]:
    import cv2

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:limit]
    return [cv2.imread(str(p)) for p in paths]


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean": round(statistics.fmean(values), 3), "median": round(statistics.median(values), 3),
            "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 3),
            "std": round(statistics.pstdev(values), 3)}


def time_precision(weights: Path, images: list[np.ndarray], settings: dict, half: bool, device: str) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(imgsz=settings["imgsz"], conf=settings["conf"], iou=settings["iou"],
                  max_det=settings["max_det"], quantize=16 if half else None, device=device, verbose=False)
    for i in range(settings["warmup"]):
        model.predict(images[i % len(images)], **kwargs)
    sync = torch.cuda.synchronize if torch.cuda.is_available() and device != "cpu" else (lambda: None)

    stages = {"preprocess": [], "inference": [], "postprocess": [], "end_to_end": []}
    for _ in range(settings["repeats"]):
        for image in images:
            sync()
            start = time.perf_counter()
            result = model.predict(image, **kwargs)[0]
            sync()
            stages["end_to_end"].append((time.perf_counter() - start) * 1000)
            for key in ("preprocess", "inference", "postprocess"):
                stages[key].append(result.speed[key])
    summary = {stage: summarize(values) for stage, values in stages.items()}
    summary["fps_end_to_end"] = round(1000 / summary["end_to_end"]["mean"], 1)
    summary["samples"] = len(stages["end_to_end"])
    return summary


def benchmark(weights: Path, image_dir: Path, settings: dict, out_dir: Path, device: str) -> dict:
    from aquadet.env import environment

    if settings["batch"] != 1:
        raise ValueError("Latency protocol is defined for batch 1")
    images = load_images(image_dir, settings["images"])
    payload = {"weights": str(weights), "settings": settings, "size": model_size(weights, settings["imgsz"]),
               "environment_before": environment(), "latency_ms": {}}
    for precision in settings["precisions"]:
        if precision == "fp16" and device == "cpu":
            continue
        print(f"[benchmark] {weights.name} {precision}: {settings['warmup']} warm-up, "
              f"{len(images)}x{settings['repeats']} timed")
        payload["latency_ms"][precision] = time_precision(weights, images, settings, precision == "fp16", device)
        print(f"[benchmark] {precision}: inference {payload['latency_ms'][precision]['inference']['mean']:.2f} ms, "
              f"end-to-end {payload['latency_ms'][precision]['end_to_end']['mean']:.2f} ms")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
