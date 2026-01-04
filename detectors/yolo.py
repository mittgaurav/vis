"""
Generic YOLO detector wrapper for Ultralytics models.

Provides:
- load_yolo_from_config(config, device)
- yolo_detect_frame(model, image, detector_config)

So trackers only need to call these helpers.
"""

import numpy as np
from typing import Any, Dict


def load_yolo_from_config(detector_config: Dict[str, Any], device: str):
    """
    Load a YOLO model given a detector config and device.

    detector_config expects keys:
      - model_name: str, e.g. "yolo11n", "yolo12s"
      - conf_threshold: float (optional, default 0.3)
      - params: dict with
          - filter_class: int or None (bird class id)
          - detect_all_classes: bool
          - nms_threshold: float
          - img_size: int
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics") from e

    model_name = detector_config["model_name"]

    print(f"Loading {model_name}...")
    model = YOLO(f"{model_name}.pt")
    model.to(device)
    print(f"Model loaded on {device}")

    # Unpack detector params with defaults
    conf_threshold = detector_config.get("conf_threshold", 0.3)
    params = detector_config.get("params", {}) or {}
    filter_class = params.get("filter_class", 14)  # default bird class
    detect_all_classes = params.get("detect_all_classes", False)
    nms_threshold = params.get("nms_threshold", 0.45)
    img_size = params.get("img_size", 640)

    print(f"Detector config: conf={conf_threshold}, " f"nms={nms_threshold}, img_size={img_size}, " f"filter_class={filter_class}, detect_all={detect_all_classes}")

    # Return model and a small config dict the trackers can store
    runtime_cfg = {
        "conf_threshold": conf_threshold,
        "filter_class": filter_class,
        "detect_all_classes": detect_all_classes,
        "nms_threshold": nms_threshold,
        "img_size": img_size,
    }
    return model, runtime_cfg


def yolo_detect_frame(
    model: Any,
    image: np.ndarray,
    runtime_cfg: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """
    Run YOLO detection on a single BGR frame.

    Args:
        model: Ultralytics YOLO model.
        image: BGR image (H, W, 3).
        runtime_cfg: dict returned by load_yolo_from_config.
        device: "cpu" or "cuda:0".

    Returns:
        detections: numpy array (N, 5) of [x, y, w, h, confidence]
    """
    conf_threshold = runtime_cfg["conf_threshold"]
    nms_threshold = runtime_cfg["nms_threshold"]
    img_size = runtime_cfg["img_size"]
    filter_class = runtime_cfg["filter_class"]
    detect_all_classes = runtime_cfg["detect_all_classes"]

    # Inference
    results = model(
        image,
        conf=conf_threshold,
        iou=nms_threshold,
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if detect_all_classes or filter_class is None or cls == filter_class:
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                detections.append([x, y, w, h, conf])

    if len(detections) == 0:
        return np.empty((0, 5), dtype=float)
    return np.asarray(detections, dtype=float)
