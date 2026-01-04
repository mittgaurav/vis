"""
YOLO with tile-based detection for large images with tiny objects
"""

import numpy as np
from baselines.base_tracker import BaseTracker
from trackers.sort import Sort
from detectors.yolo import load_yolo_from_config


class YOLOTiledSORT(BaseTracker):
    """YOLO with tile-based inference for small objects in large images"""

    def _initialize_detector(self):
        """Initialize YOLO"""
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_yolo_from_config(detector_config, self.device)

        # Tile parameters
        self.tile_size = detector_config.get("tile_size", 1024)
        self.overlap = detector_config.get("overlap", 256)  # Overlap between tiles

        print(f"YOLO Tiled detector: tile_size={self.tile_size}, overlap={self.overlap}")
        return self.detector

    def _initialize_tracker(self):
        """Initialize SORT"""
        tracker_params = self.config["tracker"]["params"]
        return Sort(
            max_age=tracker_params.get("max_age", 10),
            min_hits=tracker_params.get("min_hits", 1),
            iou_threshold=tracker_params.get("iou_threshold", 0.1),
        )

    def _detect_frame(self, image):
        """
        Detect using tile-based YOLO
        Process image in overlapping tiles to find small objects
        """
        h, w = image.shape[:2]
        detections = []

        stride = self.tile_size - self.overlap

        # Create tiles
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Get tile (with bounds checking)
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)

                tile = image[y:y_end, x:x_end]

                # Run YOLO on tile
                results = self.detector(tile, conf=self.detector_runtime_cfg['conf_threshold'],
                                        device=self.device, verbose=False)

                # Convert detections back to original image coordinates
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Offset by tile position
                        x1 += x
                        y1 += y
                        x2 += x
                        y2 += y

                        conf = float(box.conf[0])
                        width = x2 - x1
                        height = y2 - y1

                        detections.append([x1, y1, width, height, conf])

        # Remove duplicate detections in overlap regions (NMS)
        if len(detections) > 0:
            detections = self._nms(np.array(detections), iou_threshold=0.5)

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def _nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression to remove duplicate detections"""
        if len(detections) == 0:
            return detections

        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 0] + detections[:, 2]
        y2 = detections[:, 1] + detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return detections[keep]
