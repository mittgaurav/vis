"""
CenterTrack - End-to-end detection and tracking
Uses the official CenterTrack implementation in the parent directory
"""

import numpy as np
import time
from tqdm import tqdm
from baselines.base_tracker import BaseTracker

# Try to import CenterTrack
try:
    import sys

    # adjust path so that CenterTrack/src/lib is importable
    sys.path.append("CenterTrack/src/lib")
    from detector import Detector
    from opts import opts
except ImportError:
    Detector = None
    opts = None
    print("WARNING: CenterTrack not found. Clone it under ../CenterTrack and build it.")


class CenterTracker(BaseTracker):
    """
    CenterTrack end-to-end tracker
    """

    def _initialize_detector(self):
        """Initialize CenterTrack model"""
        if Detector is None or opts is None:
            print("CenterTrack detector not available, will run placeholder.")
            return None

        det_cfg = self.config.get("detector", {})
        model_path = det_cfg.get("model_path", "CenterTrack/mot17_half.pth")
        num_classes = det_cfg.get("num_classes", 1)
        track_thresh = det_cfg.get("track_thresh", 0.4)
        pre_thresh = det_cfg.get("pre_thresh", 0.5)
        out_thresh = det_cfg.get("out_thresh", 0.2)
        gpus = det_cfg.get("gpus", "-1")  # "-1" = CPU

        # Build arg string similar to CenterTrack demo
        arg_str = f"tracking " f"--load_model {model_path} " f"--num_classes {num_classes} " f"--track_thresh {track_thresh} " f"--pre_thresh {pre_thresh} " f"--out_thresh {out_thresh} " f"--gpus {gpus}"

        opt = opts().init(arg_str.split(" "))
        detector = Detector(opt)
        return detector

    def _initialize_tracker(self):
        """CenterTrack does detection + tracking together"""
        return None

    def _detect_frame(self, image):
        """
        Not used directly in this wrapper; track_video calls detector.run(...)
        """
        return np.empty((0, 5))

    def track_video(self, dataset, video_id):
        """
        CenterTrack end-to-end tracking
        """
        predictions = {}
        frame_times = []

        frames = list(dataset.iterate_video(video_id))
        video_name = dataset.videos[video_id]["name"]

        print(f"\nProcessing video {video_name}...")

        prev_image = None

        for frame_id, image, gt_boxes, gt_ids in tqdm(frames, desc=f"Video {video_name}", unit="frame", leave=False):
            start_time = time.time()

            if self.detector is not None:
                # CenterTrack expects (current, previous)
                ret = self.detector.run(image, prev_image)
                results = ret.get("results", {})
            else:
                results = {}

            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            # Collect predictions
            predictions[frame_id] = []
            # results is a dict: cls_id -> list of dicts
            for cls_id, objs in results.items():
                for obj in objs:
                    x1, y1, x2, y2 = obj["bbox"]
                    track_id = obj.get("tracking_id", -1)
                    score = float(obj.get("score", obj.get("conf", 1.0)))
                    w, h = x2 - x1, y2 - y1
                    predictions[frame_id].append((int(track_id), [x1, y1, w, h], score))

            prev_image = image.copy()

        stats = {
            "total_frames": len(frame_times),
            "total_time": sum(frame_times),
            "avg_fps": 1.0 / (sum(frame_times) / len(frame_times)) if len(frame_times) > 0 else 0,
            "avg_frame_time": float(np.mean(frame_times)) if len(frame_times) > 0 else 0,
            "total_detections": sum(len(p) for p in predictions.values()),
            "avg_detections_per_frame": float(np.mean([len(p) for p in predictions.values()])) if len(predictions) > 0 else 0,
        }

        return predictions, stats


if __name__ == "__main__":
    print("CenterTrack tracker ready!")
    print("Ensure the official CenterTrack repo is cloned under CenterTrack and built.")
