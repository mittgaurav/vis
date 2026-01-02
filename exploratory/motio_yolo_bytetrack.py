"""
Motion-Guided YOLO + ByteTrack Tracker
Uses RAFT flow to predict track positions and run YOLO only in search windows.
"""

import numpy as np
import torch
from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


class MotionYOLOByteTrack(BaseTracker):
    """
    Novel tracker: RAFT motion prediction + focused YOLO ROIs + ByteTrack association.
    For small fast birds: predict where they'll be, search only there.
    """

    def _initialize_detector(self):
        """Initialize YOLO + RAFT"""
        detector_config = self.config["detector"]
        self.yolo_model, self.yolo_runtime_cfg = load_yolo_from_config(detector_config, self.device)

        # Load RAFT (off-the-shelf, no training needed)
        try:
            from kornia.feature import RAFT

            self.raft = RAFT.from_pretrained("facebookresearch/raft-small").eval()
            if "cuda" in self.device:
                self.raft = self.raft.cuda()
            print("RAFT loaded for motion prediction")
        except:
            print("RAFT not available, falling back to full-frame YOLO")
            self.raft = None

        # Motion-guided params
        self.search_radius = self.config.get("motion_guided", {}).get("search_radius", 64)
        self.min_roi_dets = self.config.get("motion_guided", {}).get("min_roi_dets", 3)
        self.full_frame_prob = self.config.get("motion_guided", {}).get("full_frame_prob", 0.1)

        # Track state
        self.prev_image = None
        self.prev_predictions = []
        self.flow_cache = {}

        return self.yolo_model

    def _initialize_tracker(self):
        """ByteTrack for association"""
        tracker_params = self.config["tracker"]["params"]
        return ByteTrack(
            high_thresh=tracker_params.get("high_thresh", 0.5),
            low_thresh=tracker_params.get("low_thresh", 0.1),
            iou_threshold=tracker_params.get("iou_threshold", 0.3),
            max_age=tracker_params.get("max_age", 30),
            min_hits=tracker_params.get("min_hits", 2),
        )

    def _detect_frame(self, image):
        """Motion-guided detection"""
        if self.prev_image is None:
            # First frame: full detection
            self.prev_image = image.copy()
            full_dets = yolo_detect_frame(self.yolo_model, image, self.yolo_runtime_cfg, self.device)
            self.prev_predictions = full_dets.tolist()
            return full_dets

        # Compute RAFT optical flow if available
        if self.raft is not None:
            with torch.no_grad():
                flow = self.raft(torch.from_numpy(self.prev_image), torch.from_numpy(image))
                flow_np = flow[0][0].cpu().numpy()  # [H,W,2]
        else:
            flow_np = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)

        # Predict previous track positions using flow
        predicted_centers = []
        for prev_det in self.prev_predictions:
            x, y, w, h, score = prev_det
            cx, cy = x + w / 2, y + h / 2

            # Sample flow at track center (bilinear)
            flow_x, flow_y = self._sample_flow(flow_np, cx, cy)

            # Predicted position
            pred_cx, pred_cy = cx + flow_x, cy + flow_y
            pred_x, pred_y = pred_cx - w / 2, pred_cy - h / 2

            predicted_centers.append({"bbox": [pred_x, pred_y, w, h], "score": score * 0.9, "center": (pred_cx, pred_cy)})  # slight decay

        # Detect in ROIs around predicted positions
        roi_detections = []
        for pred in predicted_centers:
            roi_dets = self._yolo_roi_search(image, pred)
            roi_detections.extend(roi_dets)

        # Fallback: occasional full-frame scan for new objects
        if len(roi_detections) < self.min_roi_dets or np.random.random() < self.full_frame_prob:
            print("  [Fallback] Full-frame YOLO scan")
            full_dets = yolo_detect_frame(self.yolo_model, image, self.yolo_runtime_cfg, self.device)
            roi_detections.extend(full_dets)

        # Update state
        self.prev_image = image.copy()
        self.prev_predictions = roi_detections

        return np.array(roi_detections)

    def _yolo_roi_search(self, image, pred_track):
        """Run YOLO in a small ROI around predicted track position"""
        x, y, w, h = pred_track["bbox"]
        cx, cy = pred_track["center"]

        # ROI around prediction (square)
        roi_size = max(self.search_radius * 2, w * 3, h * 3)
        roi_x1 = max(0, int(cx - roi_size / 2))
        roi_y1 = max(0, int(cy - roi_size / 2))
        roi_x2 = min(image.shape[1], int(cx + roi_size / 2))
        roi_y2 = min(image.shape[0], int(cy + roi_size / 2))

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return []

        # Small, fast YOLO inference
        results = self.yolo_model(roi, imgsz=128, conf=0.1, verbose=False)  # tiny input  # low threshold in ROI

        detections = []
        for result in results:
            for box in result.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()

                # Map back to full image coordinates
                gx1, gy1 = rx1 + roi_x1, ry1 + roi_y1
                gw, gh = rx2 - rx1, ry2 - ry1

                # Boost confidence if close to prediction
                iou_pred = compute_iou([gx1, gy1, gw, gh], pred_track["bbox"])
                conf_boost = 1.0 + 2.0 * iou_pred

                detections.append([gx1, gy1, gw, gh, conf * conf_boost])

        return detections

    def _sample_flow(self, flow, cx, cy):
        """Bilinear sample flow at (cx, cy)"""
        x, y = int(cx), int(cy)
        if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
            return flow[y, x]
        return 0.0, 0.0
