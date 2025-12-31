"""
YOLO + SORT tracker implementation
Inherits from BaseTracker - only implements detector-specific logic
"""
import numpy as np
from baselines.base_tracker import BaseTracker
from trackers.sort_tracker import Sort


class YOLOSORTTracker(BaseTracker):
    """YOLO detector + SORT tracker"""

    def _initialize_detector(self):
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        detector_config = self.config['detector']
        model_name = detector_config['model_name']

        print(f"Loading {model_name}...")
        model = YOLO(f'{model_name}.pt')
        model.to(self.device)
        print(f"Model loaded on {self.device}")

        # Store detector params
        self.conf_threshold = detector_config.get('conf_threshold', 0.3)
        self.filter_class = detector_config['params'].get('filter_class', 14)  # Bird class
        self.detect_all_classes = detector_config['params'].get('detect_all_classes', False)
        self.nms_threshold = detector_config['params'].get('nms_threshold', 0.45)
        self.img_size = detector_config['params'].get('img_size', 640)

        print(
            f"Detector config: conf={self.conf_threshold}, "
            f"nms={self.nms_threshold}, img_size={self.img_size}, "
            f"filter_class={self.filter_class}, detect_all={self.detect_all_classes}"
        )

        return model

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config['tracker']['params']

        tracker = Sort(
            max_age=tracker_params.get('max_age', 1),
            min_hits=tracker_params.get('min_hits', 3),
            iou_threshold=tracker_params.get('iou_threshold', 0.3)
        )

        return tracker

    def _detect_frame(self, image):
        """
        Run YOLO detection on frame

        Args:
            image: BGR image (H, W, 3)

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        # Run inference
        results = self.detector(
            image,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        detections = []

        # Extract detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # Filter by class if specified
                if self.detect_all_classes or self.filter_class is None or cls == self.filter_class:
                    detections.append([x, y, w, h, conf])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))


# Test the tracker
if __name__ == "__main__":
    import yaml
    from utils.data_loader import SMOT4SBDataset

    # Load config
    with open('configs/yolo11n_sort.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load base config and merge
    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    # Merge configs (experiment config overrides base)
    merged_config = {**base_config, **config, 'detector': config['detector'], 'tracker': config['tracker']}

    print("Testing YOLO + SORT tracker...")
    print(f"Config: {merged_config['name']}")

    # Initialize tracker
    tracker = YOLOSORTTracker(merged_config)

    # Load dataset
    dataset = SMOT4SBDataset(
        merged_config['data']['root'],
        merged_config['data']['annotation_file']
    )

    # Test on first video
    video_ids = dataset.get_video_ids()[:1]
    for video_id in video_ids:
        predictions, stats = tracker.track_video(dataset, video_id)
        print(f"\nProcessed video {video_id}")
        print(f"Stats: {stats}")
