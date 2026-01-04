"""
Fast Motion-Filtered YOLO: Run YOLO once on full image,
then filter detections to keep only those in motion regions.
Super fast, no multiple YOLO inference calls.
"""

import numpy as np
import cv2
from baselines.base_tracker import BaseTracker
from trackers.sort import Sort
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


class MotionYOLOSORT(BaseTracker):
    """Run YOLO once, filter by motion regions"""

    def _initialize_detector(self):
        """Initialize YOLO and background subtractor"""
        detector_config = self.config["detector"]

        # YOLO - run ONCE per frame
        self.yolo_model, self.yolo_runtime_cfg = load_yolo_from_config(
            detector_config, self.device
        )

        # Background subtractor for motion filtering
        motion_config = detector_config.get("motion_detection", {})
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=motion_config.get("history", 500),
            varThreshold=motion_config.get("var_threshold", 16),
            detectShadows=False
        )

        self.motion_min_area = motion_config.get("min_area", 100)
        self.motion_max_area = motion_config.get("max_area", 2000)
        self.motion_expand = motion_config.get("expand_region", 50)

        print("Motion-Filtered YOLO initialized")
        return self.yolo_model

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config["tracker"]["params"]
        return Sort(
            max_age=tracker_params.get("max_age", 10),
            min_hits=tracker_params.get("min_hits", 1),
            iou_threshold=tracker_params.get("iou_threshold", 0.01),
        )

    def _get_motion_mask(self, image):
        """Get binary motion mask"""
        fg_mask = self.bg_subtractor.apply(image)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask

    def _detect_frame(self, image):
        """
        Super fast pipeline:
        1. Run YOLO ONCE on full image (0.15s)
        2. Filter to keep only detections in motion regions (0.05s)
        Total: ~0.2s per frame
        """

        # Stage 1: Get motion mask (fast)
        motion_mask = self._get_motion_mask(image)

        # Dilate motion mask to catch birds near motion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.motion_expand, self.motion_expand))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        # Stage 2: Run YOLO ONCE on full image (not per region!)
        detections = yolo_detect_frame(
            self.yolo_model,
            image,
            self.yolo_runtime_cfg,
            self.device
        )

        if len(detections) == 0:
            return np.empty((0, 5))

        # Stage 3: Filter detections - keep only those in motion regions
        filtered = []
        for det in detections:
            x, y, w, h, conf = det

            # Check if detection center is in motion region
            cx, cy = int(x + w/2), int(y + h/2)

            if 0 <= cy < motion_mask.shape[0] and 0 <= cx < motion_mask.shape[1]:
                if motion_mask[cy, cx] > 0:  # In motion region
                    filtered.append(det)

        return np.array(filtered) if len(filtered) > 0 else np.empty((0, 5))

    def reset(self):
        """Reset for new video"""
        if self.tracker:
            self.tracker.reset()

        motion_config = self.config["detector"].get("motion_detection", {})
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=motion_config.get("history", 500),
            varThreshold=motion_config.get("var_threshold", 16),
            detectShadows=False
        )


if __name__ == "__main__":
    print("Motion-Filtered YOLO SORT ready!")
