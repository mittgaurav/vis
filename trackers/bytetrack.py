import numpy as np
np.bool = bool
from scipy.optimize import linear_sum_assignment

from trackers.sort import KalmanBoxTracker, iou_batch  # reuse


def _hungarian_match(iou_matrix, iou_threshold):
    if iou_matrix.size == 0:
        return (np.empty((0, 2), dtype=int), np.arange(iou_matrix.shape[0]), np.arange(iou_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = []
    unmatched_rows = list(range(iou_matrix.shape[0]))
    unmatched_cols = list(range(iou_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] < iou_threshold:
            continue
        matches.append([r, c])
        if r in unmatched_rows:
            unmatched_rows.remove(r)
        if c in unmatched_cols:
            unmatched_cols.remove(c)
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.asarray(matches, dtype=int)
    return matches, np.asarray(unmatched_rows, dtype=int), np.asarray(unmatched_cols, dtype=int)


class ByteTrack:
    """
    Simplified ByteTrack-style tracker for [x,y,w,h,score] detections.
    """

    def __init__(self, high_thresh=0.5, low_thresh=0.1, iou_threshold=0.3, max_age=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: (N,5) [x,y,w,h,score]
        """
        self.frame_count += 1

        # split high / low score detections
        if detections.shape[0] > 0 and detections.shape[1] == 5:
            scores = detections[:, 4]
            high_mask = scores >= self.high_thresh
            low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)
            dets_high = detections[high_mask, :4]
            dets_low = detections[low_mask, :4]
        else:
            dets_high = detections[:, :4] if detections.shape[0] > 0 else np.empty((0, 4))
            dets_low = np.empty((0, 4))

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

        # if no tracks yet, initialize from high-score detections
        if len(self.trackers) == 0:
            for i in range(dets_high.shape[0]):
                self.trackers.append(KalmanBoxTracker(dets_high[i]))
            return self._gather_outputs()

        # 1) associate high-score detections to tracks
        iou_mat = iou_batch(dets_high, trks) if dets_high.shape[0] > 0 and trks.shape[0] > 0 else np.zeros((0, 0))
        matches_h, unmatched_h, unmatched_trks = _hungarian_match(iou_mat, self.iou_threshold)

        # update matched tracks with high-score detections
        for m in matches_h:
            self.trackers[m[1]].update(dets_high[m[0]])

        # 2) associate low-score detections to remaining unmatched tracks
        if dets_low.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            trks_remain = trks[unmatched_trks]
            iou_mat2 = iou_batch(dets_low, trks_remain)
            matches_l, unmatched_l, unmatched_trks2 = _hungarian_match(iou_mat2, self.iou_threshold)

            # update those tracks with low-score dets
            for i, (d_idx, t_idx) in enumerate(matches_l):
                global_trk_idx = unmatched_trks[t_idx]
                self.trackers[global_trk_idx].update(dets_low[d_idx])

            # remaining unmatched tracks indices update
            unmatched_trks = unmatched_trks[unmatched_trks2]

        # 3) create new tracks for unmatched high-score detections
        for idx in unmatched_h:
            self.trackers.append(KalmanBoxTracker(dets_high[idx]))

        # 4) age & remove dead tracks, then output
        return self._gather_outputs()

    def _gather_outputs(self):
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

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
