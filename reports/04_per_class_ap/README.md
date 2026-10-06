# Per-class AP tables

- **`per_class_ap.md`**: the tables for the report. They give validation AP@[.5:.95] and AP50 per class for YOLOv8n (`last.pt` and `best.pt`) and YOLOE-26s (bare / synonym / description), the per-class change relative to YOLOE bare names, and a separate YOLOv8n **test** table. The test table is not comparable with the validation rows.
- **`per_class_ap_valid.csv`**: the same validation numbers in long format (model, class, GT boxes, AP, AP50).
- Per-class AP as a function of training-set size (Setup 3) is in `../03_setup3_data_efficiency/tables.md`.

All validation rows come from one COCOeval call per model in `scripts/analysis/build_tables.py`, on the same 127 images and 909 boxes. The "mean" column equals COCO AP, because COCO AP is the unweighted mean over classes. So starfish, with 27 boxes, weighs as much as fish, with 459.

## Validation AP@[.5:.95] (×100), the core table

| Model | fish (459) | jellyfish (155) | penguin (104) | puffin (74) | shark (57) | starfish (27) | stingray (33) | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n last.pt | 43.5 | 51.9 | 30.5 | 25.0 | 44.7 | 55.3 | 59.9 | 44.4 |
| YOLOE bare | 19.6 | 12.8 | 11.2 | 6.2 | 5.4 | 17.7 | 18.7 | 13.1 |
| YOLOE synonym | 19.8 | **3.0** | 11.2 | 6.1 | 5.4 | **25.4** | 18.5 | 12.8 |
| YOLOE description | 1.6 | 3.1 | 3.3 | 0.1 | 2.5 | **27.2** | 2.6 | 5.8 |

## Observations

1. **The synonym effect is confined to the two changed prompts.** `sea star` raises starfish AP by 7.7 points, while `sea jelly` lowers jellyfish by 9.9. The five unchanged classes move by at most 0.3 points. The overall −0.3 is the net of two opposite effects, so it should not be summarised as "synonyms don't matter".
2. **Descriptions help only starfish** (+9.5, its best result under any wording). They collapse fish (−18.0) and stingray (−16.2). The overlays show fish becoming shark, penguin and starfish (112, 110 and 108 boxes; `../05_qualitative_overlays`). This is consistent with the generic fish description losing to the more specific descriptions of other classes, a hypothesis that was not tested.
3. **Both models find puffin and penguin hard.** These are YOLOv8n's two weakest classes (25.0, 30.5) and among YOLOE's weakest (6.2, 11.2). Likely contributors: they are the smallest objects relative to the image (median about 0.4 % of the image area) and appear in groups, and at least one puffin image (`val_047`) is shot through rain-spattered glass. Shark is YOLOE's weakest class (5.4) but mid-table for YOLOv8n (44.7). YOLOE labels many sharks as `fish` (41 boxes at its display threshold), while the supervised YOLOv8n rarely does (5).
4. **The rare classes are noisy.** Starfish has 27 validation boxes and stingray 33, so one image can move their AP by several points. Setup 3 gives seed spreads per class.
