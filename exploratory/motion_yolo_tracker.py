"""
Motion-Guided YOLO Tracker (Lightweight Novel Approach)
Motion detection + Multi-scale YOLO only (no DINO, no optical flow)
MUCH FASTER than full pipeline
"""
import numpy as np
import cv2

from baselines.base_tracker import BaseTracker
from trackers.sort import Sort


class MotionYOLOTracker(BaseTracker):
    """
    Lightweight novel approach:
    1. Motion detection (background subtraction)
    2. Multi-scale YOLO on motion regions only
    3. Standard SORT tracking

    Key innovation: Only run YOLO where motion is detected (efficient!)
    """

    def _initialize_detector(self):
        """Initialize motion detection and YOLO"""
        config = self.config["detector"]

        # 1. YOLO for object detection
        from ultralytics import YOLO

        yolo_config = config["yolo"]
        print(f"Loading YOLO: {yolo_config['model_name']}")
        self.yolo_model = YOLO(f"{yolo_config['model_name']}.pt")
        self.yolo_model.to(self.device)
        self.yolo_scales = yolo_config["scales"]
        self.yolo_conf = yolo_config["conf_threshold"]
        self.filter_class = yolo_config.get("filter_class", 14)

        # 2. Background subtractor for motion detection
        motion_config = config["motion_detection"]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=motion_config["history"], varThreshold=motion_config["var_threshold"], detectShadows=False)
        self.motion_min_area = motion_config["min_area"]
        self.motion_max_area = motion_config["max_area"]

        print("Motion-Guided YOLO tracker initialized!")

        return self.yolo_model

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = Sort(max_age=tracker_params["max_age"], min_hits=tracker_params["min_hits"], iou_threshold=tracker_params["iou_threshold"])

        return tracker

    def _detect_motion_regions(self, image):
        """Detect motion regions using background subtraction"""
        fg_mask = self.bg_subtractor.apply(image)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.motion_min_area < area < self.motion_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Expand region slightly
                expansion = 20
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(image.shape[1] - x, w + 2 * expansion)
                h = min(image.shape[0] - y, h + 2 * expansion)
                motion_regions.append([x, y, w, h])

        return motion_regions

    def _detect_yolo_multiscale(self, image, motion_regions=None):
        """Run YOLO at multiple scales on motion regions"""
        detections = []

        if motion_regions is None or len(motion_regions) == 0:
            # No motion, run on full image at single scale
            regions = [[0, 0, image.shape[1], image.shape[0]]]
            scales = [1.0]
        else:
            regions = motion_regions
            scales = self.yolo_scales

        for region in regions:
            x, y, w, h = [int(v) for v in region]

            # Crop region
            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            # Run YOLO at multiple scales
            for scale in scales:
                if scale != 1.0:
                    scaled_h, scaled_w = int(h * scale), int(w * scale)
                    if scaled_h == 0 or scaled_w == 0:
                        continue
                    scaled_crop = cv2.resize(crop, (scaled_w, scaled_h))
                else:
                    scaled_crop = crop

                # YOLO inference
                results = self.yolo_model(scaled_crop, conf=self.yolo_conf, device=self.device, verbose=False)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls == self.filter_class:
                            # Get box in original image coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # Scale back and offset
                            if scale != 1.0:
                                x1, y1 = x1 / scale, y1 / scale
                                x2, y2 = x2 / scale, y2 / scale

                            x1 += x
                            y1 += y
                            x2 += x
                            y2 += y

                            detections.append([x1, y1, x2 - x1, y2 - y1, conf])

        # NMS across scales
        if len(detections) > 0:
            detections = self._nms(np.array(detections))

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def _nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression"""
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

    def _detect_frame(self, image):
        """
        Complete detection pipeline:
        1. Find motion regions
        2. Run multi-scale YOLO on those regions
        """
        # Stage 1: Motion detection
        motion_regions = self._detect_motion_regions(image)

        # Stage 2: Multi-scale YOLO on motion regions
        detections = self._detect_yolo_multiscale(image, motion_regions)

        return detections

    def track_video(self, dataset, video_id):
        """Override to reset per video"""
        # Reset SORT tracker
        if self.tracker is not None:
            self.tracker.reset()

        # Reset background subtractor
        motion_config = self.config["detector"]["motion_detection"]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=motion_config["history"], varThreshold=motion_config["var_threshold"], detectShadows=False)

        return super().track_video(dataset, video_id)


if __name__ == "__main__":
    print("Motion-Guided YOLO Tracker ready!")
    print("Use: python run_baseline.py --config configs/motion_yolo_tracker.yaml")
