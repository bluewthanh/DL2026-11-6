"""Dependency-free checks for OWLv2 COCO mapping and validation-only CLI guards.

Run from repository root: python -m unittest discover -s scripts/open_vocab -p 'test_run_owlv2.py'
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# validate_coco is in scripts/data/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

# Stub validate_coco when pycocotools is not installed.
try:
    from validate_coco import CATEGORIES
except ModuleNotFoundError as exc:
    if exc.name != "pycocotools":
        raise
    module = types.ModuleType("validate_coco")
    module.CATEGORIES = {1: "fish", 2: "jellyfish", 3: "penguin", 4: "puffin",
                         5: "shark", 6: "starfish", 7: "stingray"}
    module.validate = lambda path: None
    with patch.dict(sys.modules, {"validate_coco": module}):
        from run_owlv2 import parse_args, predictions_for_image
else:
    from run_owlv2 import parse_args, predictions_for_image


class TensorStub:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class TestMapping(unittest.TestCase):
    def test_bbox_label_clamping_sort_and_max_det(self):
        result = {"boxes": TensorStub([[-2, 2, 12, 8], [1, 1, 3, 4], [2, 3, 2, 8]]),
                  "scores": TensorStub([0.7, 0.9, 0.8]),
                  "labels": TensorStub([0, 6, 3])}
        image = {"id": 42, "width": 10, "height": 9}
        found = predictions_for_image(result, image, .001, 300)
        self.assertEqual(found, [
            {"image_id": 42, "category_id": 7, "bbox": [1, 1, 2, 3], "score": .9},
            {"image_id": 42, "category_id": 1, "bbox": [0., 2, 10., 6], "score": .7},
        ])
        self.assertEqual(predictions_for_image(result, image, .001, 1), found[:1])

    def test_invalid_label_or_score_fails(self):
        image = {"id": 1, "width": 10, "height": 10}
        for label, score in ((7, .9), (0, float("nan"))):
            with self.subTest(label=label, score=score):
                result = {"boxes": TensorStub([[1, 1, 2, 2]]),
                          "scores": TensorStub([score]), "labels": TensorStub([label])}
                with self.assertRaises(ValueError):
                    predictions_for_image(result, image, .001, 300)

    def test_guardrails(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            prefix = ["--source", str(source), "--output", str(Path(root) / "new")]
            for extra in (["--coco", "data/aquarium-v2-coco/test.json"],
                          ["--limit", "0"], ["--conf", "nan"], ["--max-det", "0"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    parse_args(prefix + extra)
            self.assertEqual(parse_args(prefix + ["--limit", "10"]).limit, 10)


if __name__ == "__main__":
    unittest.main()
