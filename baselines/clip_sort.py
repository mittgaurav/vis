"""
CLIP + SORT tracker implementation
Uses CLIP for zero-shot bird detection via sliding window
"""

import numpy as np
np.bool = bool
import torch
import cv2
from PIL import Image
from baselines.base_tracker import BaseTracker
from trackers.sort import Sort


class CLIPSORT(BaseTracker):
    """CLIP detector (sliding window) + SORT tracker"""

    def _initialize_detector(self):
        """Initialize CLIP model"""
        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip not installed. Install with:\n" "  pip install open-clip-torch")

        detector_config = self.config["detector"]
        model_name = detector_config.get("model_name", "ViT-B/32")

        print(f"Loading open_clip {model_name}...")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        model = model.to(self.device)
        tokenizer = open_clip.get_tokenizer(model_name)

        print(f"CLIP loaded on {self.device}")

        # Store detector params
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.text_prompt = detector_config["params"].get("text_prompt", "a bird")
        self.threshold = detector_config.get("conf_threshold", 0.3)

        # Sliding window params
        self.window_sizes = detector_config["params"].get("window_sizes", [32, 64, 128])
        self.stride = detector_config["params"].get("stride", 16)
        self.min_area = detector_config["params"].get("min_area", 0)
        self.max_area = detector_config["params"].get("max_area", float("inf"))

        print(f"CLIP config: prompt='{self.text_prompt}', threshold={self.threshold}")
        print(f"Sliding window: sizes={self.window_sizes}, stride={self.stride}")
        print(f"Area filter: [{self.min_area}, {self.max_area}]")

        # Encode text prompt
        with torch.no_grad():
            text = self.tokenizer([self.text_prompt, "background"]).to(self.device)
            self.text_features = model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        return model

    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        tracker_params = self.config["tracker"]["params"]

        tracker = Sort(max_age=tracker_params.get("max_age", 1), min_hits=tracker_params.get("min_hits", 3), iou_threshold=tracker_params.get("iou_threshold", 0.3))

        return tracker

    def _detect_frame(self, image):
        """
        Run CLIP detection via sliding window (batched for speed)
        """
        h, w = image.shape[:2]
        detections = []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_windows = self.config["detector"]["params"].get("max_windows_per_frame", 512)
        stride = self.config["detector"]["params"].get("stride", self.stride)

        windows = []
        coords = []

        for window_size in self.window_sizes:
            for y in range(0, h - window_size + 1, stride):
                for x in range(0, w - window_size + 1, stride):
                    window = image_rgb[y : y + window_size, x : x + window_size]
                    windows.append(self.preprocess(Image.fromarray(window)))
                    coords.append((x, y, window_size, window_size))

                    if len(windows) >= max_windows:
                        break
                if len(windows) >= max_windows:
                    break
            if len(windows) >= max_windows:
                break

        if len(windows) == 0:
            return np.empty((0, 5))

        windows_tensor = torch.stack(windows).to(self.device)
        with torch.no_grad():
            image_features = self.detector.encode_image(windows_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            bird_scores = similarity[:, 0].cpu().numpy()

        # Collect detections above threshold
        for (x, y, ws, hs), score in zip(coords, bird_scores):
            if score > self.threshold:
                # ✅ NEW (critical)
                area = ws * hs
                if area < self.min_area or area > self.max_area:
                    continue

                detections.append([x, y, ws, hs, score])

        if len(detections) > 0:
            detections = self._nms(np.array(detections), iou_threshold=0.5)

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
