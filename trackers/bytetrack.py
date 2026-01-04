import numpy as np
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
        # print(f"\n=== FRAME {self.frame_count} ===")
        # print(f"Detections input: {detections.shape if len(detections) > 0 else 'EMPTY'}")

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

        # print(f"High dets: {dets_high.shape[0]} (>= {self.high_thresh})")
        # print(f"Low dets: {dets_low.shape[0]} (>= {self.low_thresh})")
        # print(f"Active trackers: {len(self.trackers)}")

        # predict existing tracks
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # FIXED: remove from BOTH trackers AND trks
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        # print(f"Valid tracks after predict: {trks.shape[0]}")

        if len(self.trackers) == 0:
            # print(f"NO TRACKERS: Init {dets_high.shape[0]} new from high")
            for i in range(dets_high.shape[0]):
                self.trackers.append(KalmanBoxTracker(dets_high[i]))
            outs = self._gather_outputs()
            # print(f"Output: {outs.shape if len(outs) > 0 else 'EMPTY'}")
            return outs

        # 1) associate high-score detections to tracks
        if dets_high.shape[0] > 0 and trks.shape[0] > 0:
            iou_mat = iou_batch(dets_high, trks)
            # print(f"High IoU mat: {iou_mat.shape}, max IoU: {iou_mat.max():.3f}")
        else:
            iou_mat = np.zeros((dets_high.shape[0], trks.shape[0]))
            # print(f"High IoU mat: {iou_mat.shape} (empty case)")

        matches_h, unmatched_h, unmatched_trks = _hungarian_match(iou_mat, self.iou_threshold)
        # print(f"High matches: {len(matches_h)}, unmatched high: {len(unmatched_h)}, unmatched trks: {len(unmatched_trks)}")

        # update matched tracks
        for m in matches_h:
            self.trackers[m[1]].update(dets_high[m[0]])

        # 2) associate low-score detections to remaining unmatched tracks
        # print(f"Low stage: {dets_low.shape[0]} low dets, {len(unmatched_trks)} unmatched trks")
        if dets_low.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            trks_remain = trks[unmatched_trks]
            iou_mat2 = iou_batch(dets_low, trks_remain)
            # print(f"Low IoU mat max: {iou_mat2.max():.3f}")
            matches_l, unmatched_l, unmatched_trks2 = _hungarian_match(iou_mat2, self.iou_threshold)
            # print(f"Low matches: {len(matches_l)}")

            for i, (d_idx, t_idx) in enumerate(matches_l):
                global_trk_idx = unmatched_trks[t_idx]
                self.trackers[global_trk_idx].update(dets_low[d_idx])
            unmatched_trks = unmatched_trks[unmatched_trks2]

        # 3) create new tracks for unmatched high-score detections
        # print(f"Creating {len(unmatched_h)} new tracks from high dets")
        for idx in unmatched_h:
            self.trackers.append(KalmanBoxTracker(dets_high[idx]))

        # 4) output
        outs = self._gather_outputs()
        # print(f"FINAL OUTPUT: {outs.shape if len(outs) > 0 else 'EMPTY'} tracks")
        return outs

    def _gather_outputs(self):
        ret = []
        for trk in self.trackers:
            # output ALL recent tracks, not just this-frame matches
            if trk.time_since_update <= self.max_age:
                d = trk.get_state()
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
