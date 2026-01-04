"""
Motion-Guided YOLO + ByteTrack Tracker
Uses RAFT flow to predict track positions and run YOLO only in search windows.
"""

import numpy as np
import torch
import cv2
from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


def compute_iou(box1, box2):
    """IoU between two boxes [x,y,w,h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)

    inter_w, inter_h = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = inter_w * inter_h
    union = w1*h1 + w2*h2 - inter

    return inter / (union + 1e-6) if union > 0 else 0.0


class RAFTYOLOByteTrack(BaseTracker):
    """
    Novel tracker: RAFT motion prediction + focused YOLO ROIs + ByteTrack association.
    For small fast birds: predict where they'll be, search only there.
    """

    def _initialize_detector(self):
        """Initialize YOLO + RAFT"""
        detector_config = self.config["detector"]
        self.yolo_model, self.yolo_runtime_cfg = load_yolo_from_config(detector_config, self.device)

        # Load RAFT with fallback (off-the-shelf, no training needed)
        self.raft = None
        self.prev_gray = None
        try:
            from torchvision.models.optical_flow import raft_small
            self.raft = raft_small(weights=None, progress=False).eval()  # No pretrained weights = tiny memory
            self.raft = self.raft.to(self.device)
            print("RAFT loaded for motion prediction (Torchvision)")
        except:
            print("RAFT unavailable, using OpenCV Farneback")

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
            if self.raft is None:
                self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            full_dets = yolo_detect_frame(self.yolo_model, image, self.yolo_runtime_cfg, self.device)
            self.prev_predictions = full_dets.tolist()
            return full_dets

        # Compute RAFT optical flow if available
        flow_np = self._get_optical_flow(image)

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

    def _get_optical_flow(self, image):
        """Get optical flow with RAFT or OpenCV fallback"""
        if self.raft is not None:
            h, w = image.shape[:2]
            # RAFT requires H,W divisible by 8 + max size 368x496
            target_h = (h // 8) * 8  # Round down to nearest multiple of 8
            target_w = (w // 8) * 8
            target_h = min(target_h, 368)  # RAFT max safe height
            target_w = min(target_w, 496)  # RAFT max safe width

            img1_small = cv2.resize(self.prev_image, (target_w, target_h))
            img2_small = cv2.resize(image, (target_w, target_h))

            img1_rgb = cv2.cvtColor(img1_small, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2_small, cv2.COLOR_BGR2RGB)
            img1_torch = torch.from_numpy(img1_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            img2_torch = torch.from_numpy(img2_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            with torch.no_grad():
                flow_list = self.raft(img1_torch, img2_torch)
                flow = flow_list[-1]  # final flow
                flow_np_small = flow[0].permute(1, 2, 0).cpu().numpy()  # [H,W,2]
                flow_np_small *= 20  # scale to pixels

                # Resize back to original size
                flow_np = cv2.resize(flow_np_small, (w, h))
            return flow_np
        else:
            # OpenCV Farneback fallback (unchanged)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)

            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            self.prev_gray = gray
            return flow

    def _yolo_roi_search(self, image, pred_track):
        """Run YOLO in a small ROI around predicted track position"""
        x, y, w, h = pred_track["bbox"]
        cx, cy = pred_track["center"]

        # ROI around prediction (square) - BIGGER
        roi_size = max(self.search_radius * 2, w * 4, h * 4)  # ↑ Bigger padding
        roi_x1 = max(0, int(cx - roi_size / 2))
        roi_y1 = max(0, int(cy - roi_size / 2))
        roi_x2 = min(image.shape[1], int(cx + roi_size / 2))
        roi_y2 = min(image.shape[0], int(cy + roi_size / 2))

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return []

        # ULTRA LOW threshold for tiny birds in ROIs
        roi_conf = self.config.get("motion_guided", {}).get("roi_conf", 0.01)
        results = self.yolo_model(roi, imgsz=256, conf=roi_conf, verbose=False)  # ↑ Bigger input too

        detections = []
        for result in results:
            if result.boxes is not None:  # Safety check
                for box in result.boxes:
                    rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()

                    # Map back to full image coordinates
                    gx1, gy1 = rx1 + roi_x1, ry1 + roi_y1
                    gw, gh = rx2 - rx1, ry2 - ry1

                    # MASSIVE confidence boost for ROI detections
                    iou_pred = compute_iou([gx1, gy1, gw, gh], pred_track["bbox"])
                    conf_boost = 1.0 + 5.0 * iou_pred  # ↑ 5x boost (was 2x)

                    detections.append([gx1, gy1, gw, gh, conf * conf_boost])

        return detections

    def _sample_flow(self, flow, cx, cy):
        """Bilinear sample flow at (cx, cy)"""
        x, y = int(cx), int(cy)
        if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
            return flow[y, x]
        return 0.0, 0.0
