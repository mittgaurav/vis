"""
Motion-based detection (Background Subtraction) + SORT
This works better for birds than YOLO alone
"""

import numpy as np
import cv2
from baselines.base_tracker import BaseTracker
from trackers.sort import Sort


class MotionSORT(BaseTracker):
    """Detect moving objects via background subtraction, track with SORT"""

    def _initialize_detector(self):
        """Initialize background subtractor"""
        # MOG2 is excellent for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )

        # Motion detection params
        self.min_area = 50  # Minimum blob size (pixels)
        self.max_area = 5000  # Maximum blob size

        print("Motion-based detector initialized")
        return self.bg_subtractor

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config["tracker"]["params"]
        return Sort(
            max_age=tracker_params.get("max_age", 10),
            min_hits=tracker_params.get("min_hits", 1),
            iou_threshold=tracker_params.get("iou_threshold", 0.1),
        )

    def _detect_frame(self, image):
        """
        Detect moving objects using background subtraction

        Args:
            image: BGR image

        Returns:
            detections: (N, 5) array of [x, y, w, h, confidence]
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)

        # Threshold to binary
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on area (normalized)
                confidence = min(1.0, area / 500.0)

                detections.append([x, y, w, h, confidence])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def reset(self):
        """Reset for new video"""
        if self.tracker:
            self.tracker.reset()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )


if __name__ == "__main__":
    print("Motion-based SORT tracker ready!")
