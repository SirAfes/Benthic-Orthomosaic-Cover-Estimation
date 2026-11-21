"""Optional heavy detectors behind the ``img_rgb -> instances`` interface (lazy imports).

Neither backend can run in the unit tests (they need torch, weights and usually a GPU); the shared
post-processing around them is tested in ``tests/test_detection.py``.

* ``load_yolo_seg``      Ultralytics YOLO segmentation weights trained on your own organism masks.
* ``load_grounded_sam``  Zero-shot Grounding DINO (text prompts) boxes refined to masks by SAM2.
                         Needs no training data, so it is the cold start and the pseudo-labeller for a
                         later YOLO-seg model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from detection import polygons_to_instances


def load_yolo_seg(weights: str, device: str = "cpu", conf: float = 0.25, imgsz: int = 1024) -> Callable:
    from ultralytics import YOLO  # lazy

    model = YOLO(weights)

    def detect(img_rgb: np.ndarray) -> list:
        res = model.predict(np.ascontiguousarray(img_rgb[..., ::-1]), conf=conf, imgsz=imgsz,   # ultralytics expects BGR
                            device=device, verbose=False)[0]
        if res.masks is None or res.boxes is None:
            return []
        return polygons_to_instances([np.asarray(p) for p in res.masks.xy],
                                     res.boxes.cls.cpu().numpy().astype(int), res.names,
                                     res.boxes.conf.cpu().numpy(), img_rgb.shape[:2])

    return detect


def load_grounded_sam(prompts: dict, sam2_config: str, sam2_checkpoint: str, device: str = "cpu",
                      dino_id: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = 0.3,
                      text_threshold: float = 0.25) -> Callable:
    """``prompts`` maps class name -> text phrase, e.g. {"Corynactis viridis": "orange anemone"}."""
    import torch  # lazy
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(dino_id)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
    sam = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint, device=device))
    phrases = {cls: ph.lower().strip() for cls, ph in prompts.items()}
    text = ". ".join(phrases.values()) + "."

    def to_class(label: str):
        label = str(label).lower()
        for cls, ph in phrases.items():
            if ph in label or label in ph:
                return cls
        return None

    def detect(img_rgb: np.ndarray) -> list:
        pil = Image.fromarray(img_rgb)
        inputs = processor(images=pil, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = dino(**inputs)
        kw = dict(text_threshold=text_threshold, target_sizes=[pil.size[::-1]])
        try:
            res = processor.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_threshold, **kw)[0]
        except TypeError:                                   # older transformers
            res = processor.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=box_threshold, **kw)[0]
        labels = res.get("text_labels", res.get("labels"))
        if len(res["boxes"]) == 0:
            return []
        sam.set_image(img_rgb)
        found = []
        for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            cls = to_class(lab)
            if cls is None:
                continue
            masks, _, _ = sam.predict(box=box, multimask_output=False)
            found.append({"segmentation": masks[0].astype(bool), "class": cls, "score": float(score)})
        return found

    return detect
