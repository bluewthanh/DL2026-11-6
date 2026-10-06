"""Per-class ground-truth statistics for the report's dataset section (Section 4.5).

Counts boxes, images containing each class and the maximum per image, plus the median box
area as a share of the image. The "side at 640 px" column converts that share to an
approximate box side after letterboxing a 3:4 image to 640 px (480 x 640), the shape of most
validation images. Reads only the derived COCO ground truth; no model is run.

    python scripts/analysis/gt_stats.py --coco data/aquarium-v2-coco/valid.json
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LETTERBOX_PIXELS = 480 * 640


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, default=Path("data/aquarium-v2-coco/valid.json"))
    args = parser.parse_args()

    gt = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {i["id"]: i for i in gt["images"]}
    shares, per_image = defaultdict(list), defaultdict(Counter)
    for a in gt["annotations"]:
        img = images[a["image_id"]]
        shares[a["category_id"]].append(a["area"] / (img["width"] * img["height"]))
        per_image[a["image_id"]][a["category_id"]] += 1

    sizes = Counter((i["width"], i["height"]) for i in gt["images"])
    print(f"{args.coco}: {len(images)} images, {len(gt['annotations'])} boxes; "
          f"most common sizes {sizes.most_common(3)}")
    print(f"images with more than one class: {sum(len(c) > 1 for c in per_image.values())}\n")
    print("| Class | Boxes | Images containing it | Max per image | Median box area (% of image) | ≈ side at 640 px |")
    print("|---|---:|---:|---:|---:|---:|")
    for cat in sorted(gt["categories"], key=lambda c: c["id"]):
        cid, values = cat["id"], shares[cat["id"]]
        counts = [c[cid] for c in per_image.values() if c[cid]]
        median = statistics.median(values)
        side = (median * LETTERBOX_PIXELS) ** 0.5
        print(f"| {cat['name']} | {len(values)} | {len(counts)} | {max(counts)} | {median * 100:.2f} | ~{side:.0f} px |")


if __name__ == "__main__":
    main()
