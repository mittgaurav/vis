import os
import shutil
from trackeval import Evaluator
from trackeval.metrics import HOTA

from utils.data_loader import convert_bbox_to_mot_format


def _write_mot(data, path):
    """Write GT or predictions to MOT txt format"""
    with open(path, "w") as f:
        for frame_id in sorted(data.keys()):
            for tid, bbox in data[frame_id]:
                x, y, w, h = convert_bbox_to_mot_format(bbox)
                f.write(f"{frame_id},{tid},{x},{y},{w},{h},1,1,1\n")


class SMOT4SBTrackEvalDataset:
    """Custom TrackEval-compatible dataset wrapper for SMOT4SB"""

    def __init__(self, tmp_dir=".trackeval_tmp", tracker_names=None):
        self.tmp_dir = tmp_dir
        self.gt_folder = os.path.join(tmp_dir, "gt")
        self.trackers_folder = os.path.join(tmp_dir, "pred")
        self.trackers = tracker_names or ["tracker"]
        self.seqs = ["seq1"]

    def get_sequences(self):
        return self.seqs

    def load_sequence(self, seq):
        # Load GT
        gt_path = os.path.join(self.gt_folder, seq, "gt.txt")
        gt = {}
        with open(gt_path, "r") as f:
            for line in f:
                frame, tid, x, y, w, h, *_ = line.strip().split(",")
                frame, tid = int(frame), int(tid)
                bbox = [float(x), float(y), float(w), float(h)]
                gt.setdefault(frame, []).append((tid, bbox))

        # Load tracker predictions
        preds = {}
        for tracker in self.trackers:
            pred_path = os.path.join(self.trackers_folder, tracker, f"{seq}.txt")
            pred_data = {}
            with open(pred_path, "r") as f:
                for line in f:
                    frame, tid, x, y, w, h, *_ = line.strip().split(",")
                    frame, tid = int(frame), int(tid)
                    bbox = [float(x), float(y), float(w), float(h)]
                    pred_data.setdefault(frame, []).append((tid, bbox))
            preds[tracker] = pred_data

        return gt, preds


def compute_hota_trackeval(gt_data, pred_data, tmp_dir=".trackeval_tmp"):
    """
    Compute HOTA using TrackEval for SMOT4SB dataset
    """

    seq = "seq1"
    tracker_name = "tracker"

    # Clean temp directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    gt_dir = os.path.join(tmp_dir, "gt", seq)
    pr_dir = os.path.join(tmp_dir, "pred", tracker_name)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pr_dir, exist_ok=True)

    _write_mot(gt_data, os.path.join(gt_dir, "gt.txt"))
    _write_mot(pred_data, os.path.join(pr_dir, f"{seq}.txt"))

    # TrackEval config
    eval_cfg = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
    }

    dataset = SMOT4SBTrackEvalDataset(tmp_dir=tmp_dir, tracker_names=[tracker_name])
    evaluator = Evaluator(eval_cfg)
    metrics = [HOTA()]

    results = evaluator.evaluate([dataset], metrics)

    return float(results[tracker_name][seq]["HOTA"]["HOTA"])
