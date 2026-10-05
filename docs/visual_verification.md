# Visual verification of YOLOE-26s on validation

- Setup: 8 random validation images (seed 0), score threshold 0.25.
  Predictions from `results/yoloe26s_valid_bare_moi/predictions.json`.
- Class names: <ground truth labels match the objects / found problems: ...>
- Image size vs COCO: <no mismatch / mismatch in ...>
- Box coordinates: <predicted boxes are aligned with objects / shifted by ...>
- Errors seen: <missed objects, false positives, which classes>
- Conclusion: <coordinates and class mapping look correct, so the YOLOE
  results can be trusted for comparison / problem found: ...>