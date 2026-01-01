"""
RF-DETR + SORT tracker implementation
RF-DETR: Refined DETR detector
"""
import numpy as np
from baselines.base_tracker import BaseTracker
from trackers.sort_tracker import Sort


class RFDETRSORTTracker(BaseTracker):
    """RF-DETR detector + SORT tracker"""

    def _initialize_detector(self):
        """Initialize RF-DETR model"""
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        detector_config = self.config['detector']
        model_name = detector_config['model_name']

        print(f"Loading {model_name}...")
        model = RTDETR(f'{model_name}.pt')
        model.to(self.device)
        print(f"Model loaded on {self.device}")

        # Store detector params
        self.conf_threshold = detector_config.get('conf_threshold', 0.3)
        self.filter_class = detector_config['params'].get('filter_class', 14)
        self.detect_all_classes = detector_config['params'].get('detect_all_classes', False)
        self.min_area = detector_config['params'].get('min_area', 0)
        self.max_area = detector_config['params'].get('max_area', float('inf'))

        print(
            f"Detector config: conf={self.conf_threshold}, "
            f"class={self.filter_class}, "
            f"area=[{self.min_area}, {self.max_area}]"
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
        Run RF-DETR detection on frame

        Args:
            image: BGR image (H, W, 3)

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        results = self.detector(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # ✅ NEW (minimal)
                area = w * h
                if area < self.min_area or area > self.max_area:
                    continue

                if (
                    (self.detect_all_classes or self.filter_class is None or cls == self.filter_class)
                    and conf >= self.conf_threshold
                ):
                    detections.append([x, y, w, h, conf])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))


if __name__ == "__main__":
    print("RF-DETR + SORT tracker ready!")
    print("Use: python run_baseline.py --config configs/rfdetr_sort.yaml")
