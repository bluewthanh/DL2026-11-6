"""Overlay ground truth (green) and YOLOE predictions (red) on a few validation images.

Example (adjust the paths to match your repo):
  python scripts/visualize_val.py \
      --images-dir data/aquarium-qlnqy-v2-yolov8/valid/images \
      --gt data/aquarium-v2-coco/<coco_val_file>.json \
      --pred results/yoloe26s_valid_bare_moi/predictions.json \
      --out figures/val_check --n 8 --score-thr 0.25
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="folder with the validation images")
    ap.add_argument("--gt", required=True, help="COCO json of the validation split")
    ap.add_argument("--pred", required=True, help="predictions json (list of boxes, COCO format)")
    ap.add_argument("--out", required=True, help="output folder for the PNG files")
    ap.add_argument("--n", type=int, default=8, help="number of images to draw")
    ap.add_argument("--score-thr", type=float, default=0.25, help="minimum prediction score to draw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = json.load(open(args.gt, encoding="utf-8"))
    preds = json.load(open(args.pred, encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in gt["categories"]}
    images = {im["id"]: im for im in gt["images"]}

    # Print the class names so you can check them
    print("Classes:", cat_name)

    random.seed(args.seed)
    # Only pick images that have predictions (useful when the run used --limit)
    pred_ids = {p["image_id"] for p in preds}
    pool = [i for i in images if i in pred_ids] or list(images)
    chosen = random.sample(pool, min(args.n, len(pool)))

    for img_id in chosen:
        info = images[img_id]
        path = os.path.join(args.images_dir, info["file_name"])
        img = Image.open(path)
        # Check that the real image size matches the COCO metadata
        if img.size != (info["width"], info["height"]):
            print(f"[WARNING] {info['file_name']}: image {img.size} != COCO {(info['width'], info['height'])}")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        # COCO bbox = [x, y, w, h]
        for a in gt["annotations"]:
            if a["image_id"] == img_id:
                x, y, w, h = a["bbox"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
                ax.text(x, y - 2, cat_name[a["category_id"]], color="lime", fontsize=8)

        for p in preds:
            if p["image_id"] == img_id and p["score"] >= args.score_thr:
                x, y, w, h = p["bbox"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2, linestyle="--"))
                ax.text(x, y + h + 10, f"{cat_name[p['category_id']]} {p['score']:.2f}", color="red", fontsize=8)

        ax.set_title(f"{info['file_name']}  (green = ground truth, red = prediction)")
        ax.axis("off")
        out_path = os.path.join(args.out, f"{img_id}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print("Saved", out_path)


if __name__ == "__main__":
    main()
