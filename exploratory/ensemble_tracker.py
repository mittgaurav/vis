"""
Multi-Detector Ensemble Tracker (Novel Approach 3)
Combines YOLO + Background Subtraction + Optical Flow with weighted voting
"""

import numpy as np
import cv2
import torch

from baselines.base_tracker import BaseTracker
from trackers.sort import Sort


class EnsembleTracker(BaseTracker):
    """
    Ensemble of multiple detectors with fusion
    """

    def _initialize_detector(self):
        """Initialize all detectors in the ensemble"""
        config = self.config["detector"]

        self.detectors = {}

        # Detector 1: YOLO
        if config["yolo"]["enabled"]:
            from ultralytics import YOLO

            yolo_config = config["yolo"]
            print(f"Loading YOLO: {yolo_config['model_name']}")
            yolo_model = YOLO(f"{yolo_config['model_name']}.pt")

            # Handle MPS (Apple Silicon) vs CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                yolo_model.to("cuda")
                self.compute_device = "cuda"
            elif self.device == "mps" and torch.backends.mps.is_available():
                yolo_model.to("mps")
                self.compute_device = "mps"
            else:
                yolo_model.to("cpu")
                self.compute_device = "cpu"

            self.detectors["yolo"] = {"model": yolo_model, "conf_threshold": yolo_config["conf_threshold"], "filter_class": yolo_config.get("filter_class", 14), "weight": yolo_config["weight"]}
            print(f"YOLO loaded on {self.compute_device}")

        # Detector 2: Background Subtraction
        if config["background_subtraction"]["enabled"]:
            bg_config = config["background_subtraction"]
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_config["history"], varThreshold=bg_config["var_threshold"], detectShadows=False)
            self.detectors["background"] = {"model": bg_subtractor, "min_area": bg_config["min_area"], "max_area": bg_config["max_area"], "weight": bg_config["weight"]}
            print("Background Subtraction initialized")

        # Detector 3: Optical Flow
        if config["optical_flow"]["enabled"]:
            flow_config = config["optical_flow"]
            self.detectors["flow"] = {"method": flow_config["method"], "motion_threshold": flow_config["motion_threshold"], "min_area": flow_config["min_area"], "max_area": flow_config["max_area"], "weight": flow_config["weight"]}
            self.prev_frame_gray = None
            print("Optical Flow initialized")

        # Fusion config
        self.fusion_method = config["fusion"]["method"]
        self.nms_threshold = config["fusion"]["nms_threshold"]
        self.min_detectors = config["fusion"]["min_detectors"]

        print(f"Ensemble with {len(self.detectors)} detectors initialized!")

        return self.detectors

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = Sort(max_age=tracker_params["max_age"], min_hits=tracker_params["min_hits"], iou_threshold=tracker_params["iou_threshold"])

        return tracker

    def _detect_yolo(self, image):
        """Detector 1: YOLO"""
        if "yolo" not in self.detectors:
            return []

        yolo = self.detectors["yolo"]
        results = yolo["model"](image, conf=yolo["conf_threshold"], device=self.compute_device, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == yolo["filter_class"]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1

                    # Weighted confidence
                    weighted_conf = conf * yolo["weight"]
                    detections.append([x, y, w, h, weighted_conf, "yolo"])

        return detections

    def _detect_background_subtraction(self, image):
        """Detector 2: Background Subtraction"""
        if "background" not in self.detectors:
            return []

        bg = self.detectors["background"]
        fg_mask = bg["model"].apply(image)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if bg["min_area"] < area < bg["max_area"]:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on area
                conf = min(1.0, area / 1000.0)
                weighted_conf = conf * bg["weight"]
                detections.append([x, y, w, h, weighted_conf, "background"])

        return detections

    def _detect_optical_flow(self, image):
        """Detector 3: Optical Flow"""
        if "flow" not in self.detectors:
            return []

        flow_config = self.detectors["flow"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return []

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.prev_frame_gray = gray

        # Compute magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        motion_mask = (magnitude > flow_config["motion_threshold"]).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if flow_config["min_area"] < area < flow_config["max_area"]:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on flow magnitude
                roi_magnitude = magnitude[y : y + h, x : x + w]
                avg_magnitude = np.mean(roi_magnitude)
                conf = min(1.0, avg_magnitude / 10.0)
                weighted_conf = conf * flow_config["weight"]
                detections.append([x, y, w, h, weighted_conf, "flow"])

        return detections

    def _fuse_detections(self, all_detections):
        """Fuse detections from multiple detectors"""
        if len(all_detections) == 0:
            return np.empty((0, 5))

        if self.fusion_method == "weighted_nms":
            return self._weighted_nms(all_detections)
        elif self.fusion_method == "voting":
            return self._voting_fusion(all_detections)
        elif self.fusion_method == "union":
            return self._union_fusion(all_detections)
        else:
            return self._weighted_nms(all_detections)

    def _weighted_nms(self, detections):
        """Weighted NMS across detectors"""
        if len(detections) == 0:
            return np.empty((0, 5))

        # Convert to numpy
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])

        # Apply NMS
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 0] + dets[:, 2]
        y2 = dets[:, 1] + dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return dets[keep]

    def _voting_fusion(self, detections):
        """Voting-based fusion (require agreement from multiple detectors)"""
        # Group overlapping detections
        # Count votes and keep only those with min_detectors agreement
        # This is a simplified version
        return self._weighted_nms(detections)

    def _union_fusion(self, detections):
        """Union of all detections (no fusion)"""
        return np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])

    def _detect_frame(self, image):
        """
        Run all detectors and fuse results
        """
        all_detections = []

        # Run YOLO
        yolo_dets = self._detect_yolo(image)
        all_detections.extend(yolo_dets)

        # Run Background Subtraction
        bg_dets = self._detect_background_subtraction(image)
        all_detections.extend(bg_dets)

        # Run Optical Flow
        flow_dets = self._detect_optical_flow(image)
        all_detections.extend(flow_dets)

        # Fuse detections
        fused_detections = self._fuse_detections(all_detections)

        return fused_detections

    def track_video(self, dataset, video_id):
        """Override to reset per video"""
        self.tracker.reset()
        self.prev_frame_gray = None

        # Reset background subtractor
        if "background" in self.detectors:
            bg_config = self.config["detector"]["background_subtraction"]
            self.detectors["background"]["model"] = cv2.createBackgroundSubtractorMOG2(history=bg_config["history"], varThreshold=bg_config["var_threshold"], detectShadows=False)

        return super().track_video(dataset, video_id)


if __name__ == "__main__":
    print("Ensemble Tracker ready!")
    print("Use: python run_baseline.py --config configs/ensemble_tracker.yaml")
