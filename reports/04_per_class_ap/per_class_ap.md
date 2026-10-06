# Per-class COCO AP, validation split (127 images, 909 boxes)

Recomputed from saved predictions with one COCOeval call per model (`scripts/analysis/build_tables.py`). AP@[.5:.95] and AP50 ×100. Ground-truth box counts in the header.

## AP@[.5:.95]

| Model | fish (459) | jellyfish (155) | penguin (104) | puffin (74) | shark (57) | starfish (27) | stingray (33) | mean (= AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n last.pt | 43.5 | 51.9 | 30.5 | 25.0 | 44.7 | 55.3 | 59.9 | 44.4 |
| YOLOv8n best.pt (val-selected) | 43.8 | 53.5 | 31.2 | 24.9 | 48.5 | 55.3 | 60.5 | 45.4 |
| YOLOE-26s bare names | 19.6 | 12.8 | 11.2 | 6.2 | 5.4 | 17.7 | 18.7 | 13.1 |
| YOLOE-26s synonym set | 19.8 | 3.0 | 11.2 | 6.1 | 5.4 | 25.4 | 18.5 | 12.8 |
| YOLOE-26s descriptions | 1.6 | 3.1 | 3.3 | 0.1 | 2.5 | 27.2 | 2.6 | 5.8 |

## AP50

| Model | fish (459) | jellyfish (155) | penguin (104) | puffin (74) | shark (57) | starfish (27) | stingray (33) | mean (= AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n last.pt | 78.5 | 92.4 | 67.1 | 54.9 | 67.9 | 79.0 | 84.1 | 74.8 |
| YOLOv8n best.pt (val-selected) | 78.5 | 93.6 | 70.4 | 53.7 | 72.1 | 81.0 | 81.6 | 75.8 |
| YOLOE-26s bare names | 38.8 | 26.3 | 26.2 | 10.9 | 6.7 | 27.2 | 24.8 | 23.0 |
| YOLOE-26s synonym set | 38.8 | 7.1 | 26.3 | 10.7 | 6.7 | 34.4 | 24.4 | 21.2 |
| YOLOE-26s descriptions | 3.3 | 6.7 | 6.3 | 0.1 | 4.8 | 32.8 | 4.2 | 8.3 |

## Change in AP@[.5:.95] relative to YOLOE-26s bare names (×100)

| Model | fish | jellyfish | penguin | puffin | shark | starfish | stingray | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n last.pt | +23.9 | +39.0 | +19.4 | +18.8 | +39.3 | +37.6 | +41.1 | +31.3 |
| YOLOv8n best.pt (val-selected) | +24.2 | +40.6 | +20.0 | +18.7 | +43.1 | +37.6 | +41.7 | +32.3 |
| YOLOE-26s synonym set | +0.2 | -9.9 | +0.0 | -0.1 | +0.0 | +7.7 | -0.3 | -0.3 |
| YOLOE-26s descriptions | -18.0 | -9.8 | -7.9 | -6.1 | -2.9 | +9.5 | -16.2 | -7.3 |

## Test split, `yolov8n_e100_s0` best.pt only (63 images, 582 boxes)

Supervised YOLOv8n only. No text-prompted model was run on test, so this row must **not** be ranked against the validation rows above.

| Model | fish (249) | jellyfish (154) | penguin (82) | puffin (35) | shark (36) | starfish (11) | stingray (15) | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8n best.pt (test) | 41.8 | 57.4 | 31.6 | 22.4 | 56.5 | 59.7 | 64.0 | 47.6 |
