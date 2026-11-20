# Baseline: hand-crafted colour/texture features on the crop library

Setup: 271 crops kept (1 junk crop of 3x3 px and 1 fish crop dropped), 16 source photos,
51 colour/gradient/LBP features, leave-one-source-photo-out, only labels seen in >= 2 photos.
13 of 30 labels are evaluable (236 crops); 17 labels occur in a single photo and cannot be tested.

| Model | Macro-F1 (95% CI, cluster bootstrap over photos) | Accuracy |
|---|---|---|
| Majority class | 0.035 | 0.297 |
| Logistic regression | 0.351 (0.237 - 0.448) | 0.432 |
| Random forest | 0.287 (0.188 - 0.367) | 0.445 |

Reading: colour and texture statistics carry some signal (about 10x the majority floor) but are weak:
per-class recall ranges from 0 (Holothuria, Lissoclinum) to 0.89 (Antedon). This is a crop-level
floor for H1, not a mosaic result; it says nothing yet about domain shift to the mosaic.

Caveats: 13 classes and 16 photos give wide intervals; crops are tight boxes around organisms,
so the task is easier than classifying arbitrary mosaic patches; crop scale is not normalised.
