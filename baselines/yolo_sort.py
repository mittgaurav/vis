"""
YOLO + SORT tracker implementation
Inherits from BaseTracker - only implements detector-specific logic
"""

from baselines.base_tracker import BaseTracker
from trackers.sort import Sort
from detectors.yolo import load_yolo_from_config, yolo_detect_frame


class YOLOSORT(BaseTracker):
    """YOLO detector + SORT tracker"""

    def _initialize_detector(self):
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_yolo_from_config(detector_config, self.device)
        return self.detector

    def _initialize_tracker(self):
        tracker_params = self.config["tracker"]["params"]
        return Sort(
            max_age=tracker_params.get("max_age", 1),
            min_hits=tracker_params.get("min_hits", 3),
            iou_threshold=tracker_params.get("iou_threshold", 0.3),
        )

    def _detect_frame(self, image):
        return yolo_detect_frame(self.detector, image, self.detector_runtime_cfg, self.device)


# Test the tracker
if __name__ == "__main__":
    import yaml
    from utils.data_loader import SMOT4SBDataset

    # Load config
    with open("configs/yolo_sort.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load base config and merge
    with open("configs/base_config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Merge configs (experiment config overrides base)
    merged_config = {**base_config, **config, "detector": config["detector"], "tracker": config["tracker"]}

    print("Testing YOLO + SORT tracker...")
    print(f"Config: {merged_config['name']}")

    # Initialize tracker
    tracker = YOLOSORT(merged_config)

    # Load dataset
    dataset = SMOT4SBDataset(merged_config["data"]["root"], merged_config["data"]["annotation_file"])

    # Test on first video
    video_ids = dataset.get_video_ids()[:1]
    for video_id in video_ids:
        predictions, stats = tracker.track_video(dataset, video_id)
        print(f"\nProcessed video {video_id}")
        print(f"Stats: {stats}")
