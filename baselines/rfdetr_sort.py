"""
RF-DETR + SORT tracker implementation
Uses shared RF-DETR detector helpers.
"""

import numpy as np
from baselines.base_tracker import BaseTracker
from trackers.sort import Sort
from detectors.rfdetr import load_rfdetr_from_config, rfdetr_detect_frame


class RFDETRSORT(BaseTracker):
    """RF-DETR detector + SORT tracker"""

    def _initialize_detector(self):
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_rfdetr_from_config(detector_config, self.device)
        return self.detector

    def _initialize_tracker(self):
        tracker_params = self.config["tracker"]["params"]
        tracker = Sort(
            max_age=tracker_params.get("max_age", 1),
            min_hits=tracker_params.get("min_hits", 3),
            iou_threshold=tracker_params.get("iou_threshold", 0.3),
        )
        return tracker

    def _detect_frame(self, image):
        return rfdetr_detect_frame(
            self.detector,
            image,
            self.detector_runtime_cfg,
            self.device,
        )
