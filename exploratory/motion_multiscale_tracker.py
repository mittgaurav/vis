"""
Motion-Guided Multi-Scale Tracker (Novel Approach 1)
Combines: Motion Detection + Multi-Scale YOLO + DINO Features + Optical Flow
"""

import numpy as np

np.bool = bool
import cv2
import torch
from scipy.optimize import linear_sum_assignment
import sys

from baselines.base_tracker import BaseTracker
from trackers.sort import KalmanBoxTracker


class EnhancedTrack:
    """Track with appearance features and optical flow"""

    def __init__(self, bbox, track_id, appearance_feature=None):
        self.kalman = KalmanBoxTracker(bbox)
        self.track_id = track_id
        self.appearance_feature = appearance_feature
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.flow_velocity = np.zeros(2)  # [vx, vy] from optical flow

    def predict(self, flow_velocity=None):
        """Predict next position using Kalman + optical flow"""
        if flow_velocity is not None:
            self.flow_velocity = flow_velocity

        # Kalman prediction
        predicted_bbox = self.kalman.predict()

        # Enhance with optical flow
        if np.any(self.flow_velocity):
            predicted_bbox[0] += self.flow_velocity[0]
            predicted_bbox[1] += self.flow_velocity[1]

        self.age += 1
        return predicted_bbox

    def update(self, bbox, appearance_feature=None):
        """Update track"""
        self.time_since_update = 0
        self.hits += 1
        self.kalman.update(bbox)

        if appearance_feature is not None:
            # EMA update of appearance feature
            if self.appearance_feature is not None:
                self.appearance_feature = 0.8 * self.appearance_feature + 0.2 * appearance_feature
            else:
                self.appearance_feature = appearance_feature


class MotionMultiScaleTracker(BaseTracker):
    """
    Novel tracking approach combining multiple techniques
    """

    def _initialize_detector(self):
        """Initialize all detection components"""
        config = self.config["detector"]

        # 1. YOLO for object detection
        from ultralytics import YOLO

        yolo_config = config["yolo"]
        print(f"Loading YOLO: {yolo_config['model_name']}")
        self.yolo_model = YOLO(f"{yolo_config['model_name']}.pt")
        self.yolo_model.to(self.device)
        self.yolo_scales = yolo_config["scales"]
        self.yolo_conf = yolo_config["conf_threshold"]
        self.filter_class = yolo_config.get("filter_class", 14)

        # 2. Background subtractor for motion detection
        motion_config = config["motion_detection"]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=motion_config["history"], varThreshold=motion_config["var_threshold"], detectShadows=False)
        self.motion_min_area = motion_config["min_area"]
        self.motion_max_area = motion_config["max_area"]

        # 3. DINO for appearance features
        dino_config = config["dino"]
        print(f"Loading DINO: {dino_config['model_name']}")

        # DINO has issues with MPS, force CPU for feature extraction
        if self.device == "mps":
            print("WARNING: DINO doesn't work well with MPS, using CPU for feature extraction")
            self.dino_device = "cpu"
        else:
            self.dino_device = self.device

        self.dino_model = torch.hub.load("facebookresearch/dinov2", dino_config["model_name"])
        self.dino_model.to(self.dino_device)
        self.dino_model.eval()
        self.dino_similarity_threshold = dino_config["similarity_threshold"]

        # 4. Optical flow (initialized per video)
        self.prev_frame_gray = None
        self.optical_flow_method = config["optical_flow"]["method"]

        print("All detection components loaded!")

        return {"yolo": self.yolo_model, "bg_subtractor": self.bg_subtractor, "dino": self.dino_model}

    def _initialize_tracker(self):
        """Initialize enhanced tracker"""
        self.tracks = []
        self.next_track_id = 1

        tracker_params = self.config["tracker"]["params"]
        self.max_age = tracker_params["max_age"]
        self.min_hits = tracker_params["min_hits"]
        self.iou_threshold = tracker_params["iou_threshold"]
        self.use_appearance = tracker_params["use_appearance"]
        self.appearance_weight = tracker_params.get("appearance_weight", 0.3)
        self.motion_weight = tracker_params.get("motion_weight", 0.7)

        return None  # We manage tracks manually

    def _detect_motion_regions(self, image):
        """Stage 1: Detect motion regions using background subtraction"""
        fg_mask = self.bg_subtractor.apply(image)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.motion_min_area < area < self.motion_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Expand region slightly for YOLO
                expansion = 20
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = w + 2 * expansion
                h = h + 2 * expansion
                motion_regions.append([x, y, w, h])

        return motion_regions

    def _detect_yolo_multiscale(self, image, motion_regions=None):
        """Stage 2: Run YOLO at multiple scales on motion regions"""
        detections = []

        if motion_regions is None or len(motion_regions) == 0:
            # No motion detected, run on full image
            regions = [[0, 0, image.shape[1], image.shape[0]]]
        else:
            regions = motion_regions

        for region in regions:
            x, y, w, h = region
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Crop region
            crop = image[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            # Run YOLO at multiple scales
            for scale in self.yolo_scales:
                if scale != 1.0:
                    scaled_h, scaled_w = int(h * scale), int(w * scale)
                    scaled_crop = cv2.resize(crop, (scaled_w, scaled_h))
                else:
                    scaled_crop = crop

                # YOLO inference
                results = self.yolo_model(scaled_crop, conf=self.yolo_conf, device=self.device, verbose=False)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls == self.filter_class:
                            # Get box in original image coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # Scale back and offset
                            if scale != 1.0:
                                x1, y1 = x1 / scale, y1 / scale
                                x2, y2 = x2 / scale, y2 / scale

                            x1 += x
                            y1 += y
                            x2 += x
                            y2 += y

                            detections.append([x1, y1, x2 - x1, y2 - y1, conf])

        # NMS across scales
        if len(detections) > 0:
            detections = self._nms(np.array(detections))

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def _extract_dino_features(self, image, detections):
        """Stage 3: Extract DINO features for each detection"""
        features = []

        for det in detections:
            x, y, w, h = [int(v) for v in det[:4]]

            # Crop and resize
            crop = image[y : y + h, x : x + w]
            if crop.size == 0:
                features.append(np.zeros(384))
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))

            # To tensor
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = crop_tensor.unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                feature = self.dino_model(crop_tensor).cpu().numpy().flatten()

            features.append(feature)

        return np.array(features)

    def _compute_optical_flow(self, image):
        """Stage 4: Compute optical flow"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return None

        # Compute dense optical flow using Farneback
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.prev_frame_gray = gray
        return flow

    def _get_flow_at_bbox(self, flow, bbox):
        """Get average flow velocity within bbox"""
        if flow is None:
            return np.zeros(2)

        x, y, w, h = [int(v) for v in bbox]
        roi_flow = flow[y : y + h, x : x + w]

        if roi_flow.size == 0:
            return np.zeros(2)

        avg_flow = np.mean(roi_flow, axis=(0, 1))
        return avg_flow

    def _associate_detections_to_tracks(self, detections, features, flow):
        """Associate detections to tracks using appearance + motion"""
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        # Predict track positions
        predicted_bboxes = []
        for track in self.tracks:
            flow_velocity = self._get_flow_at_bbox(flow, track.kalman.get_state()) if flow is not None else None
            pred_bbox = track.predict(flow_velocity)
            predicted_bboxes.append(pred_bbox)

        predicted_bboxes = np.array(predicted_bboxes)

        # Compute cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)))

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                # IoU cost
                iou = self._compute_iou(det[:4], predicted_bboxes[j])
                iou_cost = 1 - iou

                # Appearance cost
                if self.use_appearance and track.appearance_feature is not None and features is not None:
                    similarity = np.dot(features[i], track.appearance_feature)
                    similarity = (similarity + 1) / 2  # Normalize to [0, 1]
                    appearance_cost = 1 - similarity
                else:
                    appearance_cost = 0.5

                # Combined cost
                cost_matrix[i, j] = self.motion_weight * iou_cost + self.appearance_weight * appearance_cost

        # Hungarian algorithm
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter by threshold
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.7:  # Threshold
                    matches.append([r, c])

            matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)

        unmatched_dets = [d for d in range(len(detections)) if d not in matches[:, 0]]
        unmatched_trks = [t for t in range(len(self.tracks)) if t not in matches[:, 1]]

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi = max(x1, x2)
        yi = max(y1, y2)
        xi_max = min(x1 + w1, x2 + w2)
        yi_max = min(y1 + h1, y2 + h2)

        inter_w = max(0, xi_max - xi)
        inter_h = max(0, yi_max - yi)
        inter = inter_w * inter_h

        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)

    def _nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return detections

        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 0] + detections[:, 2]
        y2 = detections[:, 1] + detections[:, 3]
        scores = detections[:, 4]

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

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return detections[keep]

    def _detect_frame(self, image):
        """
        Complete detection pipeline
        """
        # Stage 1: Motion detection
        motion_regions = self._detect_motion_regions(image)

        # Stage 2: Multi-scale YOLO on motion regions
        detections = self._detect_yolo_multiscale(image, motion_regions)

        # Stage 3: Extract DINO features
        features = None
        if len(detections) > 0 and self.use_appearance:
            features = self._extract_dino_features(image, detections)

        # Stage 4: Compute optical flow
        flow = self._compute_optical_flow(image)

        # Store for tracking
        self._current_features = features
        self._current_flow = flow

        return detections

    def _track_frame(self, detections):
        """
        Enhanced tracking with appearance and optical flow
        """
        features = self._current_features
        flow = self._current_flow

        # Associate detections to tracks
        matches, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(detections, features, flow)

        # Update matched tracks
        for m in matches:
            det_idx, trk_idx = m
            feature = features[det_idx] if features is not None else None
            self.tracks[trk_idx].update(detections[det_idx][:4], feature)

        # Create new tracks
        for i in unmatched_dets:
            feature = features[i] if features is not None else None
            track = EnhancedTrack(detections[i][:4], self.next_track_id, feature)
            self.tracks.append(track)
            self.next_track_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Update time since update for unmatched tracks
        for i in unmatched_trks:
            if i < len(self.tracks):
                self.tracks[i].time_since_update += 1

        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                bbox = track.kalman.get_state()
                active_tracks.append(np.concatenate([bbox, [track.track_id]]))

        return np.array(active_tracks) if len(active_tracks) > 0 else np.empty((0, 5))

    def track_video(self, dataset, video_id):
        """Override to reset per video"""
        # Reset trackers
        self.tracks = []
        self.next_track_id = 1
        self.prev_frame_gray = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=self.config["detector"]["motion_detection"]["history"], varThreshold=self.config["detector"]["motion_detection"]["var_threshold"], detectShadows=False)

        # Call parent track_video
        return super().track_video(dataset, video_id)


if __name__ == "__main__":
    print("Motion-Guided Multi-Scale Tracker ready!")
    print("Use: python run_baseline.py --config configs/motion_multiscale_tracker.yaml")
