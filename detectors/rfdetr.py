"""
Generic RF-DETR (Ultralytics RTDETR) detector wrapper.

Provides:
- load_rfdetr_from_config(detector_config, device)
- rfdetr_detect_frame(model, image, runtime_cfg, device)

Returns detections as [x, y, w, h, confidence].
"""

from typing import Any, Dict
import numpy as np


def load_rfdetr_from_config(detector_config: Dict[str, Any], device: str):
    """
    Load an RF-DETR / RTDETR model given detector config and device.

    detector_config expects keys:
      - model_name: str, e.g. "rtdetr-l", "rtdetr-b"
      - conf_threshold: float (optional, default 0.3)
      - params: dict with
          - filter_class: int or None (bird class id)
          - detect_all_classes: bool
          - min_area: float
          - max_area: float
    """
    try:
        from ultralytics import RTDETR
    except ImportError as e:
        raise ImportError("ultralytics not installed. Install with: pip install ultralytics") from e

    model_name = detector_config["model_name"]

    print(f"Loading {model_name}...")
    model = RTDETR(f"{model_name}.pt")
    model.to(device)
    print(f"Model loaded on {device}")

    conf_threshold = detector_config.get("conf_threshold", 0.3)
    params = detector_config.get("params", {}) or {}
    filter_class = params.get("filter_class", 14)
    detect_all_classes = params.get("detect_all_classes", False)
    min_area = params.get("min_area", 0.0)
    max_area = params.get("max_area", float("inf"))

    print(f"Detector config: conf={conf_threshold}, " f"class={filter_class}, area=[{min_area}, {max_area}], " f"detect_all={detect_all_classes}")

    runtime_cfg = {
        "conf_threshold": conf_threshold,
        "filter_class": filter_class,
        "detect_all_classes": detect_all_classes,
        "min_area": min_area,
        "max_area": max_area,
    }
    return model, runtime_cfg


def rfdetr_detect_frame(
    model: Any,
    image: np.ndarray,
    runtime_cfg: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """
    Run RF-DETR detection on a single BGR frame.

    Args:
        model: Ultralytics RTDETR model.
        image: BGR image (H, W, 3).
        runtime_cfg: dict returned by load_rfdetr_from_config.
        device: "cpu" or "cuda:0".

    Returns:
        detections: numpy array (N, 5) of [x, y, w, h, confidence]
    """
    conf_threshold = runtime_cfg["conf_threshold"]
    filter_class = runtime_cfg["filter_class"]
    detect_all_classes = runtime_cfg["detect_all_classes"]
    min_area = runtime_cfg["min_area"]
    max_area = runtime_cfg["max_area"]

    results = model(
        image,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            area = w * h
            if area < min_area or area > max_area:
                continue

            if (detect_all_classes or filter_class is None or cls == filter_class) and conf >= conf_threshold:
                detections.append([x, y, w, h, conf])

    if len(detections) == 0:
        return np.empty((0, 5), dtype=float)
    return np.asarray(detections, dtype=float)
