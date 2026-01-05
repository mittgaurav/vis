"""
YOLO with tile-based detection + ByteTrack tracker
Combines spatial tiling for better small object detection with
ByteTrack's two-stage matching for robust association
"""

import numpy as np
from baselines.base_tracker import BaseTracker
from trackers.bytetrack import ByteTrack
from detectors.yolo import load_yolo_from_config


class YOLOTiledByteTrack(BaseTracker):
    """YOLO tile-based detection + ByteTrack tracking"""

    def _initialize_detector(self):
        """Initialize YOLO with tile parameters"""
        detector_config = self.config["detector"]
        self.detector, self.detector_runtime_cfg = load_yolo_from_config(
            detector_config, self.device
        )

        # Tile parameters
        self.tile_size = detector_config.get("tile_size", 512)
        self.overlap = detector_config.get("overlap", 128)

        # Merge parameters
        self.merge_config = detector_config.get("merge", {})
        self.merge_enabled = self.merge_config.get("enabled", True)
        self.merge_nms_threshold = self.merge_config.get("nms_threshold", 0.5)

        print(
            f"YOLO Tiled detector initialized: "
            f"tile_size={self.tile_size}, overlap={self.overlap}"
        )
        return self.detector

    def _initialize_tracker(self):
        """Initialize ByteTrack with small-object parameters"""
        tracker_params = self.config["tracker"]["params"]

        tracker = ByteTrack(
            high_thresh=tracker_params.get("high_thresh", 0.5),
            low_thresh=tracker_params.get("low_thresh", 0.05),
            iou_threshold=tracker_params.get("iou_threshold", 0.01),
            max_age=tracker_params.get("max_age", 10),
            min_hits=tracker_params.get("min_hits", 1),
        )

        print(
            f"ByteTrack initialized: "
            f"high_thresh={tracker.high_thresh}, low_thresh={tracker.low_thresh}, "
            f"iou_threshold={tracker.iou_threshold}"
        )
        return tracker

    def _detect_frame(self, image):
        """
        Tile-based YOLO detection with ByteTrack-compatible output

        Returns:
            detections: (N, 5) array [x, y, w, h, confidence]
        """
        h, w = image.shape[:2]
        detections = []

        stride = self.tile_size - self.overlap

        # Process tiles
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Get tile with bounds checking
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)

                tile = image[y:y_end, x:x_end]

                # Run YOLO on tile
                results = self.detector(
                    tile,
                    conf=self.detector_runtime_cfg['conf_threshold'],
                    device=self.device,
                    verbose=False
                )

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

        # Remove duplicate detections in overlap regions via NMS
        if len(detections) > 0:
            detections = np.array(detections)
            if self.merge_enabled:
                detections = self._nms(
                    detections,
                    iou_threshold=self.merge_nms_threshold
                )
        else:
            detections = np.empty((0, 5))

        return detections

    def _nms(self, detections, iou_threshold=0.5):
        """
        Non-maximum suppression to remove duplicate detections
        in overlap regions between tiles

        Args:
            detections: (N, 5) array [x, y, w, h, confidence]
            iou_threshold: IoU threshold for suppression

        Returns:
            detections: (M, 5) array after NMS
        """
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
