"""
RAFT-DINO Tracker (Novel Approach 2)
Dense optical flow + DINO appearance features
Simpler and faster than Approach 1
"""
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('..')

from baselines.base_tracker import BaseTracker


class AppearanceTrack:
    """Track with appearance feature matching"""

    def __init__(self, bbox, track_id, appearance_feature):
        self.bbox = bbox  # [x, y, w, h]
        self.track_id = track_id
        self.appearance_feature = appearance_feature
        self.velocity = np.zeros(2)  # [vx, vy]
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self):
        """Predict next position using velocity"""
        self.bbox[0] += self.velocity[0]
        self.bbox[1] += self.velocity[1]
        self.age += 1
        return self.bbox

    def update(self, bbox, appearance_feature):
        """Update track"""
        # Update velocity
        self.velocity[0] = bbox[0] - self.bbox[0]
        self.velocity[1] = bbox[1] - self.bbox[1]

        # Update bbox
        self.bbox = bbox

        # Update appearance feature (EMA)
        self.appearance_feature = 0.7 * self.appearance_feature + 0.3 * appearance_feature

        self.time_since_update = 0
        self.hits += 1


class RAFTDINOTracker(BaseTracker):
    """
    RAFT-DINO Tracker: Optical flow + DINO appearance
    """

    def _initialize_detector(self):
        """Initialize optical flow and DINO"""
        config = self.config['detector']

        # DINO for appearance features
        dino_config = config['dino']
        print(f"Loading DINO: {dino_config['model_name']}")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_config['model_name'])

        # Handle MPS (Apple Silicon) vs CUDA vs CPU
        # DINO has issues with MPS, force CPU
        if self.device == 'mps':
            print("WARNING: DINO doesn't work well with MPS, using CPU for feature extraction")
            self.dino_model.to('cpu')
            self.compute_device = 'cpu'
        elif self.device == 'cuda' and torch.cuda.is_available():
            self.dino_model.to('cuda')
            self.compute_device = 'cuda'
        else:
            self.dino_model.to('cpu')
            self.compute_device = 'cpu'

        print(f"DINO loaded on {self.compute_device}")
        self.dino_model.eval()

        # Optical flow config
        flow_config = config['optical_flow']
        self.flow_method = flow_config['method']
        self.motion_threshold = flow_config['motion_threshold']
        self.min_area = flow_config['min_area']
        self.max_area = flow_config['max_area']

        self.prev_frame_gray = None

        print("RAFT-DINO tracker initialized!")

        return self.dino_model

    def _initialize_tracker(self):
        """Initialize appearance-based tracker"""
        self.tracks = []
        self.next_track_id = 1

        tracker_params = self.config['tracker']['params']
        self.max_age = tracker_params['max_age']
        self.min_hits = tracker_params['min_hits']
        self.appearance_threshold = tracker_params['appearance_threshold']
        self.motion_weight = tracker_params.get('motion_weight', 0.5)
        self.appearance_weight = tracker_params.get('appearance_weight', 0.5)

        return self

    def reset(self):
        self.tracks = []
        self.next_track_id = 1

    def _compute_optical_flow(self, image):
        """Compute dense optical flow"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return None

        if self.flow_method == 'farneback':
            # Farneback optical flow (faster, works on CPU/MPS)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        else:
            # For RAFT, would need to implement separately
            # Fallback to Farneback
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        self.prev_frame_gray = gray
        return flow

    def _detect_motion_blobs(self, flow):
        """Detect motion blobs from optical flow"""
        if flow is None:
            return []

        # Compute flow magnitude
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

        # Threshold
        motion_mask = (magnitude > self.motion_threshold).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on flow magnitude in region
                roi_magnitude = magnitude[y:y+h, x:x+w]
                avg_magnitude = np.mean(roi_magnitude)
                confidence = min(1.0, avg_magnitude / 10.0)

                detections.append([x, y, w, h, confidence])

        return detections

    def _extract_dino_features(self, image, detections):
        """Extract DINO features for detections"""
        if len(detections) == 0:
            return []

        features = []

        for det in detections:
            x, y, w, h = [int(v) for v in det[:4]]

            # Crop and resize
            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                features.append(np.zeros(384))
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))

            # To tensor
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
            crop_tensor = crop_tensor.unsqueeze(0).to(self.compute_device)

            # Extract features
            with torch.no_grad():
                feature = self.dino_model(crop_tensor).cpu().numpy().flatten()
                # Normalize
                feature = feature / (np.linalg.norm(feature) + 1e-8)

            features.append(feature)

        return np.array(features)

    def _detect_frame(self, image):
        """
        Detect using optical flow
        """
        # Compute optical flow
        flow = self._compute_optical_flow(image)

        # Detect motion blobs
        detections = self._detect_motion_blobs(flow)

        if len(detections) == 0:
            self._current_features = []
            return np.empty((0, 5))

        # Extract DINO features
        features = self._extract_dino_features(image, detections)

        # Store for tracking
        self._current_features = features

        return np.array(detections)

    def _associate_detections_to_tracks(self, detections, features):
        """Associate detections to tracks using appearance"""
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(self.tracks))

        # Predict track positions
        predicted_bboxes = []
        for track in self.tracks:
            pred_bbox = track.predict()
            predicted_bboxes.append(pred_bbox)

        predicted_bboxes = np.array(predicted_bboxes)

        # Compute cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)))

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                # Motion cost (IoU-based)
                iou = self._compute_iou(det[:4], predicted_bboxes[j])
                motion_cost = 1 - iou

                # Appearance cost (cosine distance)
                if len(features) > 0:
                    similarity = np.dot(features[i], track.appearance_feature)
                    appearance_cost = 1 - similarity
                else:
                    appearance_cost = 0.5

                # Combined cost
                cost_matrix[i, j] = (self.motion_weight * motion_cost +
                                    self.appearance_weight * appearance_cost)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter by threshold
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.8:  # Threshold
                matches.append([r, c])

        matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        unmatched_dets = [d for d in range(len(detections)) if d not in matches[:, 0]]
        unmatched_trks = [t for t in range(len(self.tracks)) if t not in matches[:, 1]]

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU"""
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

    def _track_frame(self, detections):
        """Track using appearance features"""
        features = self._current_features

        # Associate
        matches, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, features
        )

        # Update matched tracks
        for m in matches:
            det_idx, trk_idx = m
            self.tracks[trk_idx].update(detections[det_idx][:4], features[det_idx])

        # Create new tracks
        for i in unmatched_dets:
            if len(features) > 0:
                track = AppearanceTrack(detections[i][:4], self.next_track_id, features[i])
                self.tracks.append(track)
                self.next_track_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Update unmatched tracks
        for i in unmatched_trks:
            if i < len(self.tracks):
                self.tracks[i].time_since_update += 1

        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                bbox = track.bbox
                active_tracks.append(np.concatenate([bbox, [track.track_id]]))

        return np.array(active_tracks) if len(active_tracks) > 0 else np.empty((0, 5))

    def track_video(self, dataset, video_id):
        """Override to reset per video"""
        self.tracks = []
        self.next_track_id = 1
        self.prev_frame_gray = None

        return super().track_video(dataset, video_id)


if __name__ == "__main__":
    print("RAFT-DINO Tracker ready!")
    print("Use: python run_baseline.py --config configs/raft_dino_tracker.yaml")
