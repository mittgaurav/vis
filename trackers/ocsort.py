import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.sort import KalmanBoxTracker  # reuse existing KF


def iou(bboxes1, bboxes2):
    """
    Compute IoU between two sets of bounding boxes
    bb_test: (N, 4) array of [x, y, w, h]
    bb_gt: (M, 4) array of [x, y, w, h]
    Returns: (N, M) array of IoU values
    """
    b1 = np.expand_dims(bboxes1, 1)
    b2 = np.expand_dims(bboxes2, 0)
    x11, y11, w1, h1 = b1[..., 0], b1[..., 1], b1[..., 2], b1[..., 3]
    x21, y21, w2, h2 = b2[..., 0], b2[..., 1], b2[..., 2], b2[..., 3]
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    xx1 = np.maximum(x11, x21)
    yy1 = np.maximum(y11, y21)
    xx2 = np.minimum(x12, x22)
    yy2 = np.minimum(y12, y22)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def associate_ocsort(detections, trackers, iou_threshold=0.3, lambda_dist=0.98):
    """
    OC-SORT-style association:
    cost = lambda * (1 - IoU) + (1 - lambda) * normalized_center_distance
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    iou_mat = iou(detections, trackers)

    # center distance (L2 on centers, normalized by image diagonal ~ assume 1 for small birds)
    det_centers = detections[:, :2] + detections[:, 2:4] / 2.0  # (N,2)
    trk_centers = trackers[:, :2] + trackers[:, 2:4] / 2.0  # (M,2)
    dc = np.linalg.norm(det_centers[:, None, :] - trk_centers[None, :, :], axis=-1)
    # normalize by max distance in this frame to keep ~[0,1]
    if dc.size > 0:
        dc = dc / (dc.max() + 1e-6)

    cost = lambda_dist * (1.0 - iou_mat) + (1.0 - lambda_dist) * dc

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] < iou_threshold:
            continue
        matches.append([r, c])
        if r in unmatched_dets:
            unmatched_dets.remove(r)
        if c in unmatched_trks:
            unmatched_trks.remove(c)

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.asarray(matches, dtype=int)

    return matches, np.asarray(unmatched_dets, dtype=int), np.asarray(unmatched_trks, dtype=int)


class OCSort:
    """
    Lightweight OC-SORT-style tracker.
    Same interface as Sort: update(detections) -> [x,y,w,h,track_id]
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, lambda_dist=0.98):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lambda_dist = lambda_dist
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: (N,5) [x,y,w,h,score] or (N,4)
        """
        self.frame_count += 1

        # predict existing tracks
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        if detections.shape[0] == 0:
            # no detections: just age tracks and remove dead ones
            ret = []
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                d = trk.get_state()
                if (trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            if len(ret) > 0:
                return np.concatenate(ret)
            return np.empty((0, 5))

        if detections.shape[1] == 5:
            dets = detections[:, :4]
        else:
            dets = detections

        matched, unmatched_dets, unmatched_trks = associate_ocsort(dets, trks, self.iou_threshold, self.lambda_dist)

        # update matched
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # output active tracks
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
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
