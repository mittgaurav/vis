"""
SORT: Simple Online and Realtime Tracking
Implementation based on https://github.com/abewley/sort
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space
    State: [x, y, w, h, vx, vy, vw, vh] (position + velocity)
    """

    count = 0

    def __init__(self, bbox):
        """
        Initialize tracker with initial bounding box
        bbox: [x, y, w, h]
        """
        # Define constant velocity model: 8 states, 4 measurements
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])  # x = x + vx  # y = y + vy  # w = w + vw  # h = h + vh  # vx = vx  # vy = vy  # vw = vw  # vh = vh

        # Measurement matrix (we only observe position, not velocity)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])

        # Measurement noise
        self.kf.R *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocity
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with bbox
        self.kf.x[:4] = bbox.reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

    def predict(self):
        """Predict next state"""
        if self.kf.x[2] + self.kf.x[6] <= 0:  # width + velocity <= 0
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:  # height + velocity <= 0
            self.kf.x[7] = 0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.kf.x[:4].flatten())
        return self.history[-1]

    def get_state(self):
        """Get current bounding box estimate [x, y, w, h]"""
        return self.kf.x[:4].flatten()


def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between two sets of bounding boxes
    bb_test: (N, 4) array of [x, y, w, h]
    bb_gt: (M, 4) array of [x, y, w, h]
    Returns: (N, M) array of IoU values
    """
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, 0)  # (1, M, 4)

    # Convert to [x1, y1, x2, y2]
    bb_test_x2 = bb_test[..., 0] + bb_test[..., 2]
    bb_test_y2 = bb_test[..., 1] + bb_test[..., 3]
    bb_gt_x2 = bb_gt[..., 0] + bb_gt[..., 2]
    bb_gt_y2 = bb_gt[..., 1] + bb_gt[..., 3]

    # Intersection
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test_x2, bb_gt_x2)
    yy2 = np.minimum(bb_test_y2, bb_gt_y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    # Union
    area_test = bb_test[..., 2] * bb_test[..., 3]
    area_gt = bb_gt[..., 2] * bb_gt[..., 3]
    union = area_test + area_gt - intersection

    iou = intersection / (union + 1e-6)
    return iou


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to tracked objects using Hungarian algorithm

    Returns:
        matched_indices: (N, 2) array of [detection_idx, tracker_idx]
        unmatched_detections: array of detection indices
        unmatched_trackers: array of tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    # Compute IoU matrix
    iou_matrix = iou_batch(detections, trackers)

    # Hungarian algorithm for optimal assignment
    # We want to maximize IoU, so use negative for minimization
    if iou_matrix.max() > iou_threshold:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.column_stack([row_ind, col_ind])

        # Filter out matches below threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                continue
            matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        unmatched_detections = [d for d in range(len(detections)) if d not in matches[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matches[:, 1]]
    else:
        matches = np.empty((0, 2), dtype=int)
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT tracker
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: numpy array of shape (N, 5) where each row is [x, y, w, h, score]
                       or (N, 4) if no scores

        Returns:
            numpy array of shape (M, 5) where each row is [x, y, w, h, track_id]
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove trackers with NaN predictions
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        # Match detections to trackers
        if detections.shape[1] == 5:
            dets = detections[:, :4]  # Remove scores for matching
        else:
            dets = detections

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # Output all tracks that were updated this frame
        ret = []
        for trk in self.trackers:
            # Only output if track was matched/updated THIS frame
            if trk.time_since_update == 0:
                d = trk.get_state()
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

        # Remove dead trackers (age out old tracks)
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def reset(self):
        """Reset tracker for new video sequence"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0


# Example usage
if __name__ == "__main__":
    # Test SORT tracker
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    # Simulate detections over 5 frames
    print("Testing SORT tracker:")
    for frame in range(5):
        # Simulate moving bird detections
        if frame < 3:
            detections = np.array([[100 + frame * 10, 200 + frame * 5, 20, 15, 0.9], [300, 400, 18, 12, 0.85]])  # Bird moving right and down  # Static bird
        else:
            # First bird disappears
            detections = np.array([[300, 400, 18, 12, 0.85]])

        tracks = tracker.update(detections)
        print(f"Frame {frame}: {len(detections)} detections -> {len(tracks)} tracks")
        for track in tracks:
            x, y, w, h, track_id = track
            print(f"  Track {int(track_id)}: bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")

    print("\nSORT tracker ready!")
