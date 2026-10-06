All rows: 127 validation images, 909 GT boxes, pycocotools COCOeval bbox, maxDets 100, ×100. Inference for every model: imgsz 640, conf 0.001, NMS IoU 0.7, max_det 300.

| Model | Regime | Labelled Aquarium train images | AP | AP50 | AP75 | APs | APm | APl | AR100 | Detections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n last.pt | supervised fine-tuning (COCO-pretrained) | 448 | 44.4 | 74.8 | 44.9 | 3.6 | 20.8 | 47.6 | 57.3 | 14136 |
| YOLOv8n best.pt (val-selected) | supervised fine-tuning (COCO-pretrained) | 448 | 45.4 | 75.8 | 46.0 | 3.4 | 20.2 | 48.5 | 57.9 | 15329 |
| YOLOE-26s bare names | zero-shot text prompts | 0 | 13.1 | 23.0 | 11.8 | 1.0 | 6.2 | 14.7 | 35.8 | 16563 |
| YOLOE-26s synonym set | zero-shot text prompts | 0 | 12.8 | 21.2 | 12.1 | 1.0 | 3.5 | 14.3 | 34.4 | 15777 |
| YOLOE-26s descriptions | zero-shot text prompts | 0 | 5.8 | 8.3 | 5.6 | 0.0 | 2.7 | 6.0 | 26.9 | 19861 |
