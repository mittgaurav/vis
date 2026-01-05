"""
RT-DETR + ByteTrack tracker implementation
Uses shared RT-DETR detector helpers.
"""

from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.rtdetr import load_rtdetr_from_config, rtdetr_detect_frame


class RTDETRByteTrack(BaseTracker):
    """RT-DETR detector + ByteTrack tracker"""

    def _initialize_detector(self):
        """Initialize RT-DETR detector"""
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_rtdetr_from_config(
            detector_config, self.device
        )
        return self.detector

    def _initialize_tracker(self):
        """Initialize ByteTrack tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = ByteTrack(
            high_thresh=tracker_params.get("high_thresh", 0.08),
            low_thresh=tracker_params.get("low_thresh", 0.001),
            iou_threshold=tracker_params.get("iou_threshold", 0.01),
            max_age=tracker_params.get("max_age", 10),
            min_hits=tracker_params.get("min_hits", 1),
        )

        print(
            f"ByteTrack initialized: "
            f"high_thresh={tracker.high_thresh}, low_thresh={tracker.low_thresh}, "
            f"iou_threshold={tracker.iou_threshold}"
        )
        return tracker

    def _detect_frame(self, image):
        """
        Run RT-DETR detection on the full frame
        Returns [x, y, w, h, confidence] for ByteTrack
        """
        return rtdetr_detect_frame(
            self.detector,
            image,
            self.detector_runtime_cfg,
            self.device,
        )
