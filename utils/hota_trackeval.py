import os
import shutil
import numpy as np
from trackeval import Evaluator
from trackeval.metrics import HOTA

# ---------------------------------------------------------
# IoU utility
# ---------------------------------------------------------

def iou_xywh(boxA, boxB):
    """IoU for [x, y, w, h] boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - inter
    return 0.0 if union == 0 else inter / union


# ---------------------------------------------------------
# MOT writer (ROBUST TO MISSING SCORES)
# ---------------------------------------------------------

def write_mot(data, path, with_score=False):
    """
    data:
        GT   -> {frame: [(id, bbox)]}
        Pred -> {frame: [(id, bbox)] OR [(id, bbox, score)]}
    """
    with open(path, "w") as f:
        for frame in sorted(data.keys()):
            for item in data[frame]:
                if with_score:
                    if len(item) == 3:
                        tid, bbox, score = item
                    else:
                        tid, bbox = item
                        score = 1.0
                else:
                    tid, bbox = item
                    score = 1.0

                x, y, w, h = bbox
                f.write(f"{frame},{tid},{x},{y},{w},{h},{score},-1,-1,-1\n")


class SMOT4SBTrackEvalDataset:
    """Minimal TrackEval-compatible dataset (single sequence, single class)"""

    def __init__(self, tmp_dir, tracker_names):
        self.tmp_dir = tmp_dir
        self.gt_folder = os.path.join(tmp_dir, "gt")
        self.tracker_folder = os.path.join(tmp_dir, "pred")
        self.trackers = tracker_names
        self.seqs = ["seq1"]

        # ---- required flags for eval.py ----
        # no class-combination logic for a single 'bird' class
        self.should_classes_combine = False
        self.use_super_categories = False
        self.super_categories = {}   # only used if use_super_categories is True

    # ---- Required API ----

    def get_name(self):
        return "SMOT4SB"

    def get_sequences(self):
        return self.seqs

    def get_eval_info(self):
        # trackers, sequences, list of classes
        return self.trackers, self.seqs, ["bird"]

    # eval.py calls this when writing results
    def get_output_fol(self, tracker):
        out_dir = os.path.join(self.tmp_dir, "results", tracker)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # eval.py calls this for printing tables
    def get_display_name(self, tracker):
        return tracker

    # ---- Load MOT files ----

    def load_sequence(self, seq):
        # GT
        gt = {}
        with open(os.path.join(self.gt_folder, seq, "gt.txt")) as f:
            for line in f:
                frame, tid, x, y, w, h, *_ = line.strip().split(",")
                frame, tid = int(frame), int(tid)
                bbox = [float(x), float(y), float(w), float(h)]
                gt.setdefault(frame, []).append((tid, bbox))

        # Predictions
        preds = {}
        for tracker in self.trackers:
            preds[tracker] = {}
            with open(os.path.join(self.tracker_folder, tracker, f"{seq}.txt")) as f:
                for line in f:
                    frame, tid, x, y, w, h, score, *_ = line.strip().split(",")
                    frame, tid = int(frame), int(tid)
                    bbox = [float(x), float(y), float(w), float(h)]
                    preds[tracker].setdefault(frame, []).append(
                        (tid, bbox, float(score))
                    )

        return gt, preds

    def get_raw_seq_data(self, tracker, seq):
        gt, preds = self.load_sequence(seq)
        # TrackEval expects (gt, pred) order for most built-in datasets; you swapped
        # it when calling compute_hota_trackeval, so keep this consistent with
        # your get_preprocessed_seq_data.
        return preds[tracker], gt

    # ---- HOTA preprocessing ----
    def get_preprocessed_seq_data(self, raw_data, cls):
        preds, gt = raw_data

        frames = sorted(set(gt.keys()) | set(preds.keys()))

        # ---- build global ID maps ----
        gt_id_set = set()
        tr_id_set = set()

        for objs in gt.values():
            for tid, _ in objs:
                gt_id_set.add(tid)

        for objs in preds.values():
            for tid, _, _ in objs:
                tr_id_set.add(tid)

        gt_id_map = {tid: i for i, tid in enumerate(sorted(gt_id_set))}
        tr_id_map = {tid: i for i, tid in enumerate(sorted(tr_id_set))}

        # ---- outputs ----
        similarity_scores = []
        gt_ids = []
        tracker_ids = []
        gt_dets = []
        tracker_dets = []
        tracker_scores = []

        num_gt_dets = 0
        num_tracker_dets = 0

        for f in frames:
            gt_objs = gt.get(f, [])
            tr_objs = preds.get(f, [])

            gt_ids_f = np.array([gt_id_map[tid] for tid, _ in gt_objs], dtype=int)
            tr_ids_f = np.array([tr_id_map[tid] for tid, _, _ in tr_objs], dtype=int)

            gt_ids.append(gt_ids_f)
            tracker_ids.append(tr_ids_f)

            gt_boxes = np.array([bbox for _, bbox in gt_objs], dtype=float)
            tr_boxes = np.array([bbox for _, bbox, _ in tr_objs], dtype=float)

            gt_dets.append(gt_boxes)
            tracker_dets.append(tr_boxes)

            tracker_scores.append(
                np.array([score for _, _, score in tr_objs], dtype=float)
            )

            num_gt_dets += len(gt_objs)
            num_tracker_dets += len(tr_objs)

            sim = np.zeros((len(gt_objs), len(tr_objs)), dtype=np.float32)
            for i, (_, gbox) in enumerate(gt_objs):
                for j, (_, tbox, _) in enumerate(tr_objs):
                    sim[i, j] = iou_xywh(gbox, tbox)

            similarity_scores.append(sim)

        num_timesteps = len(frames)

        return {
            "similarity_scores": similarity_scores,
            "gt_ids": gt_ids,
            "tracker_ids": tracker_ids,
            "gt_dets": gt_dets,
            "tracker_dets": tracker_dets,
            "tracker_scores": tracker_scores,
            "num_gt_ids": len(gt_id_map),
            "num_tracker_ids": len(tr_id_map),
            "num_gt_dets": num_gt_dets,
            "num_tracker_dets": num_tracker_dets,
            "num_timesteps": num_timesteps,
        }


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def compute_hota_trackeval(gt_data, pred_data, tmp_dir=".trackeval_tmp"):
    """
    gt_data   : {frame: [(id, bbox)]}
    pred_data : {frame: [(id, bbox)] OR [(id, bbox, score)]}
    """

    tracker_name = "tracker"
    seq = "seq1"

    # clean tmp dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(os.path.join(tmp_dir, "gt", seq), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "pred", tracker_name), exist_ok=True)

    # write MOT files
    write_mot(gt_data, os.path.join(tmp_dir, "gt", seq, "gt.txt"))
    write_mot(
        pred_data,
        os.path.join(tmp_dir, "pred", tracker_name, f"{seq}.txt"),
        with_score=True,
    )

    # dataset + metric
    dataset = SMOT4SBTrackEvalDataset(tmp_dir, [tracker_name])
    hota_metric = HOTA()

    evaluator = Evaluator({
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "PLOT_CURVES": False,
    })

    # NOTE: evaluate returns (results_dict, messages_dict)
    output_res, output_msg = evaluator.evaluate([dataset], [hota_metric])

    dataset_name = dataset.get_name()
    cls_name = "bird"  # from get_eval_info()
    seq_res = output_res[dataset_name][tracker_name][seq][cls_name]

    # metric_name from the metric object (e.g. "HOTA")
    metric_name = hota_metric.get_name()
    if metric_name not in seq_res:
        # try to auto-detect the HOTA metric key
        candidates = [k for k in seq_res.keys() if "hota" in k.lower()]
        if not candidates:
            raise KeyError(
                f"No HOTA-like metric key found for sequence '{seq}'. "
                f"Available metric keys at class level: {list(seq_res.keys())}"
            )
        metric_name = candidates[0]

    metric_dict = seq_res[metric_name]

    # pick the submetric you want
    if "HOTA" in metric_dict:
        hota_vec = metric_dict["HOTA"]
        hota = float(hota_vec.mean())
    elif "HOTA(0)" in metric_dict:
        hota = float(metric_dict["HOTA(0)"])
    else:
        sub_candidates = [k for k in metric_dict.keys() if k.upper().startswith("HOTA")]
        if not sub_candidates:
            raise KeyError(
                f"No HOTA submetric found in metric '{metric_name}'. "
                f"Available submetrics: {list(metric_dict.keys())}"
            )
        sub_key = sub_candidates[0]
        value = metric_dict[sub_key]
        try:
            hota = float(value.mean())
        except Exception:
            hota = float(value)

    return hota
