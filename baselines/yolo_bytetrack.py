"""
YOLO + ByteTrack-style tracker implementation
Uses shared YOLO detector helpers to avoid duplication.
"""

from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


class YOLOByteTrack(BaseTracker):
    """YOLO detector + ByteTrack-style tracker"""

    def _initialize_detector(self):
        """Initialize YOLO model via shared helper"""
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_yolo_from_config(detector_config, self.device)
        return self.detector

    def _initialize_tracker(self):
        """Initialize ByteTrack-style tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = ByteTrack(
            high_thresh=tracker_params.get("high_thresh", 0.5),
            low_thresh=tracker_params.get("low_thresh", 0.1),
            iou_threshold=tracker_params.get("iou_threshold", 0.3),
            max_age=tracker_params.get("max_age", 30),
            min_hits=tracker_params.get("min_hits", 3),
        )
        return tracker

    def _detect_frame(self, image):
        """
        Run YOLO detection on frame and return [x, y, w, h, confidence]
        """
        return yolo_detect_frame(
            self.detector,
            image,
            self.detector_runtime_cfg,
            self.device,
        )

if __name__ == "__main__":
    import yaml
    from utils.data_loader import SMOT4SBDataset

    with open("configs/yolo_bytetrack.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("configs/base_config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    merged_config = {
        **base_config,
        **config,
        "detector": config["detector"],
        "tracker": config["tracker"],
    }

    print("Testing YOLO + OC-SORT tracker...")
    print(f"Config: {merged_config['name']}")

    tracker = YOLOByteTrack(merged_config)

    dataset = SMOT4SBDataset(merged_config["data"]["root"], merged_config["data"]["annotation_file"])

    video_ids = dataset.get_video_ids()[:1]
    for video_id in video_ids:
        predictions, stats = tracker.track_video(dataset, video_id)
        print(f"\nProcessed video {video_id}")
        print(f"Stats: {stats}")
