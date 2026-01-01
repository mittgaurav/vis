"""
YOLO + ByteTrack-style tracker implementation
Uses shared YOLO detector helpers to avoid duplication.
"""

from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


class YOLOByteTrack(BaseTracker):
    """YOLO detector + ByteTrack-style tracker"""

    def _initialize_detector(self):
        """Initialize YOLO model via shared helper"""
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_yolo_from_config(detector_config, self.device)
        return self.detector

    def _initialize_tracker(self):
        """Initialize ByteTrack-style tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = ByteTrack(
            high_thresh=tracker_params.get("high_thresh", 0.5),
            low_thresh=tracker_params.get("low_thresh", 0.1),
            iou_threshold=tracker_params.get("iou_threshold", 0.3),
            max_age=tracker_params.get("max_age", 30),
            min_hits=tracker_params.get("min_hits", 3),
        )
        return tracker

    def _detect_frame(self, image):
        """
        Run YOLO detection on frame and return [x, y, w, h, confidence]
        """
        return yolo_detect_frame(
            self.detector,
            image,
            self.detector_runtime_cfg,
            self.device,
        )
