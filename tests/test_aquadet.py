"""Run: python -m unittest discover -s tests"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aquadet import config as cfg
from aquadet import report
from aquadet.benchmark import summarize
from aquadet.dataset import KNOWN_INVALID, check_policy, filter_label_lines
from aquadet.evaluate import xyxy_to_coco


class ConfigTests(unittest.TestCase):
    def test_every_model_config_loads_with_identical_protocol(self):
        hashes = {cfg.load(m)["protocol_hash"] for m in cfg.available_models()}
        self.assertEqual(len(hashes), 1, "model files must not change the shared protocol")
        self.assertTrue({"yolov8n", "yolo11n", "yolov10n"} <= set(cfg.available_models()))

    def test_seed_does_not_change_protocol_but_epochs_do(self):
        base = cfg.load("yolov8n")
        seeded = cfg.load("yolov8n", ["train.seed=3"])
        custom = cfg.load("yolov8n", ["train.epochs=5"])
        self.assertEqual(base["protocol_hash"], seeded["protocol_hash"])
        self.assertTrue(seeded["protocol_matches_base"])
        self.assertNotEqual(base["protocol_hash"], custom["protocol_hash"])
        self.assertFalse(custom["protocol_matches_base"])
        self.assertEqual(cfg.default_run_name(custom), "yolov8n_e5_s0_custom")
        self.assertEqual(cfg.default_run_name(seeded), "yolov8n_e100_s3")

    def test_smoke_is_named_and_flagged(self):
        smoke = cfg.load("yolo11n", smoke=True)
        self.assertEqual(smoke["train"]["epochs"], 1)
        self.assertEqual(cfg.default_run_name(smoke), "yolo11n_e1_s0_smoke")

    def test_override_parsing_and_validation(self):
        self.assertEqual(cfg.parse_override("train.lr0=0.01"), ("train.lr0", 0.01))
        self.assertEqual(cfg.parse_override("eval.splits=[val]"), ("eval.splits", ["val"]))
        for bad in ("epochs=3", "train.epochs"):
            with self.assertRaises(ValueError):
                cfg.parse_override(bad)
        with self.assertRaises(ValueError):
            cfg.load("yolov8n", ["train.not_a_key=1"])
        with self.assertRaises(ValueError):
            cfg.load("yolov99n")


class DatasetPolicyTests(unittest.TestCase):
    def test_only_exact_known_records_are_dropped(self):
        rel, line, cls, xywh = sorted(KNOWN_INVALID)[0]
        lines = ["0 0.5 0.5 0.1 0.1"] * (line - 1) + [" ".join([cls, *xywh]), "4 0.5 0.5 0.2 0.2", ""]
        kept, omitted = filter_label_lines("\n".join(lines), rel, KNOWN_INVALID)
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]["line"], line)
        self.assertEqual(len(kept), line)
        kept, omitted = filter_label_lines("\n".join(lines), "test/labels/other.txt", KNOWN_INVALID)
        self.assertEqual(omitted, [])

    def test_unexpected_audit_findings_are_refused(self):
        source = Path("/export")
        audit = {"passed": False, "errors": ["x"], "invalid_annotations": [
            {"file": "/export/test/labels/new.txt", "line": 1, "class_id": "0", "normalized_xywh": ["0.5", "0.5", "0", "0"]}]}
        with self.assertRaises(ValueError):
            check_policy(audit, source, "omit-known-zero-boxes")
        with self.assertRaises(ValueError):
            check_policy(audit, source, "strict")
        self.assertEqual(check_policy({"passed": True, "errors": [], "invalid_annotations": []}, source, "strict"), set())


class ConversionTests(unittest.TestCase):
    def test_xyxy_to_coco_xywh(self):
        self.assertEqual(xyxy_to_coco([10, 20, 110, 70]), [10, 20, 100, 50])

    def test_latency_summary(self):
        stats = summarize([1.0, 2.0, 3.0, 4.0, 10.0])
        self.assertEqual(stats["mean"], 4.0)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["p90"], 4.0)


def fake_summary(model: str, seed: int, ap: float, device: str = "GPU A", protocol: str = "abc") -> dict:
    per_class = {c: {"AP": ap, "AP50": ap, "gt_boxes": 1} for c in report.CLASSES}
    coco = {"AP": ap, "AP50": ap + 0.1, "AP75": ap, "per_class": per_class}
    return {"run": f"{model}_s{seed}", "model": model, "seed": seed, "epochs": 100, "smoke": False,
            "protocol_hash": protocol, "protocol_matches_base": True, "overrides": {},
            "train": {"wall_time_s": 600, "best_epoch": 90},
            "eval": {"val": {"coco": coco, "ultralytics": {}},
                     "test": {"coco": coco, "ultralytics": {"P": 0.5, "R": 0.5, "mAP50": 0.5, "mAP50-95": 0.3}}},
            "benchmark": {"size": {"params_M": 3.0, "gflops": 8.0},
                          "latency_ms": {"fp32": {"inference": {"mean": 5.0}, "end_to_end": {"mean": 7.0}}}},
            "environment": {"device": device, "ultralytics": "x", "torch": "y"}}


class CompareTests(unittest.TestCase):
    def run_compare(self, summaries: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "runs").mkdir()
            for s in summaries:
                (tmp / "runs" / f"{s['run']}.json").write_text(json.dumps(s))
            with patch.object(report, "REPORTS_DIR", tmp), patch.object(report, "SUMMARY_DIR", tmp / "runs"), \
                    patch.object(report, "ROOT", tmp):
                path = report.compare()
            self.assertTrue((tmp / "benchmark.csv").is_file())
            return path.read_text()

    def test_seeds_are_aggregated_and_best_is_bold(self):
        text = self.run_compare([fake_summary("yolov8n", 0, 0.40), fake_summary("yolov8n", 1, 0.50),
                                 fake_summary("yolo11n", 0, 0.30)])
        self.assertIn("| yolov8n | 2 |", text)
        self.assertIn("**45.0 ± 7.1**", text)
        self.assertNotIn("Warnings", text)

    def test_mixed_devices_and_protocols_warn(self):
        text = self.run_compare([fake_summary("yolov8n", 0, 0.4, device="GPU A", protocol="p1"),
                                 fake_summary("yolo11n", 0, 0.4, device="GPU B", protocol="p2")])
        self.assertIn("different devices", text)
        self.assertIn("different protocols", text)


if __name__ == "__main__":
    unittest.main()
