"""Validate the derived Aquarium v2 COCO ground truth without editing any files.

Run: python scripts/validate_coco.py --coco-dir data/aquarium-v2-coco
Requires: python -m pip install pycocotools
"""

import argparse
import math
from pathlib import Path

from pycocotools.coco import COCO


CATEGORIES = {
    1: "fish", 2: "jellyfish", 3: "penguin", 4: "puffin",
    5: "shark", 6: "starfish", 7: "stingray",
}
SPLITS = ("train", "valid", "test")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def unique_ids(records: list, kind: str) -> set[int]:
    ids = set()
    for index, item in enumerate(records):
        require(isinstance(item, dict), f"{kind}[{index}] is not an object")
        identifier = item.get("id")
        require(type(identifier) is int and identifier > 0,
                f"{kind}[{index}] has an invalid id: {identifier!r}")
        require(identifier not in ids, f"duplicate {kind} id: {identifier}")
        ids.add(identifier)
    return ids


def validate(path: Path) -> tuple[int, int]:
    # COCO indexes by ID, so inspect its original arrays before trusting the index:
    # duplicate IDs would otherwise be silently overwritten.
    coco = COCO(str(path))
    data = coco.dataset
    require(isinstance(data, dict), "COCO root must be an object")
    for key in ("images", "annotations", "categories"):
        require(isinstance(data.get(key), list), f"missing or non-list {key}")
    images, annotations, categories = (data[key] for key in ("images", "annotations", "categories"))
    image_ids = unique_ids(images, "image")
    annotation_ids = unique_ids(annotations, "annotation")
    category_ids = unique_ids(categories, "category")
    require(category_ids == set(CATEGORIES), f"category IDs {sorted(category_ids)} != 1..7")
    for category in categories:
        require(category.get("name") == CATEGORIES[category["id"]],
                f"incorrect name for category {category['id']}: {category.get('name')!r}")

    dimensions = {}
    for image in images:
        width, height = image.get("width"), image.get("height")
        require(number(width) and width > 0 and number(height) and height > 0,
                f"image {image['id']} has invalid dimensions")
        require(isinstance(image.get("file_name"), str) and image["file_name"],
                f"image {image['id']} has no file_name")
        dimensions[image["id"]] = (width, height)

    for ann in annotations:
        label = f"annotation {ann['id']}"
        image_id, category_id = ann.get("image_id"), ann.get("category_id")
        require(type(image_id) is int and image_id in image_ids,
                f"{label} refers to a nonexistent image: {image_id!r}")
        require(type(category_id) is int and category_id in category_ids,
                f"{label} has an unknown category: {category_id!r}")
        bbox = ann.get("bbox")
        require(isinstance(bbox, list) and len(bbox) == 4 and all(number(v) for v in bbox),
                f"{label} has an invalid bbox: {bbox!r}")
        x, y, w, h = bbox
        image_w, image_h = dimensions[image_id]
        require(w > 0 and h > 0 and x >= -1e-6 and y >= -1e-6
                and x + w <= image_w + 1e-6 and y + h <= image_h + 1e-6,
                f"{label} has a nonpositive or out-of-image bbox: {bbox!r}")
        area = ann.get("area")
        require(number(area) and area > 0 and math.isclose(area, w * h, rel_tol=1e-8, abs_tol=1e-6),
                f"{label} has invalid area: {area!r} (bbox area {w * h})")

    require(set(coco.getImgIds()) == image_ids and len(coco.imgs) == len(images),
            "pycocotools image index disagrees with JSON")
    require(set(coco.getAnnIds()) == annotation_ids and len(coco.anns) == len(annotations),
            "pycocotools annotation index disagrees with JSON")
    require(set(coco.getCatIds()) == category_ids, "pycocotools category index disagrees with JSON")
    return len(images), len(annotations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", type=Path, default=Path("data/aquarium-v2-coco"))
    args = parser.parse_args()
    totals = [0, 0]
    for split in SPLITS:
        path = args.coco_dir / f"{split}.json"
        try:
            counts = validate(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.exit(1, f"FAIL {path}: {exc}\n")
        print(f"PASS {path}: {counts[0]} images, {counts[1]} annotations")
        totals[0] += counts[0]
        totals[1] += counts[1]
    print(f"TOTAL: {totals[0]} images, {totals[1]} annotations; all COCO checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
