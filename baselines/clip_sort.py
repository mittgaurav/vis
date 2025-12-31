"""
CLIP + SORT tracker implementation
Uses CLIP for zero-shot bird detection via sliding window
"""
import numpy as np
import torch
import cv2
from PIL import Image
from baselines.base_tracker import BaseTracker
from trackers.sort_tracker import Sort


class CLIPSORTTracker(BaseTracker):
    """CLIP detector (sliding window) + SORT tracker"""

    def _initialize_detector(self):
        """Initialize CLIP model"""
        try:
            import clip
        except ImportError:
            raise ImportError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

        detector_config = self.config['detector']
        model_name = detector_config.get('model_name', 'ViT-B/32')

        print(f"Loading CLIP {model_name}...")
        model, preprocess = clip.load(model_name, device=self.device)
        print(f"CLIP loaded on {self.device}")

        # Store detector params
        self.preprocess = preprocess
        self.text_prompt = detector_config['params'].get('text_prompt', 'a bird')
        self.threshold = detector_config.get('conf_threshold', 0.3)

        # Sliding window params
        self.window_sizes = detector_config['params'].get('window_sizes', [32, 64, 128])
        self.stride = detector_config['params'].get('stride', 16)

        print(f"CLIP config: prompt='{self.text_prompt}', threshold={self.threshold}")
        print(f"Sliding window: sizes={self.window_sizes}, stride={self.stride}")

        # Encode text prompt
        with torch.no_grad():
            text = clip.tokenize([self.text_prompt, "background"]).to(self.device)
            self.text_features = model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

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
        Run CLIP detection via sliding window

        Args:
            image: BGR image (H, W, 3)

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        h, w = image.shape[:2]
        detections = []

        # Convert BGR to RGB for CLIP
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Multi-scale sliding window
        for window_size in self.window_sizes:
            for y in range(0, h - window_size + 1, self.stride):
                for x in range(0, w - window_size + 1, self.stride):
                    # Extract window
                    window = image_rgb[y:y + window_size, x:x + window_size]

                    # Preprocess for CLIP
                    window_pil = Image.fromarray(window)
                    window_tensor = self.preprocess(window_pil).unsqueeze(0).to(self.device)

                    # Encode image
                    with torch.no_grad():
                        image_features = self.detector.encode_image(window_tensor)
                        image_features /= image_features.norm(dim=-1, keepdim=True)

                        # Compute similarity
                        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                        bird_score = similarity[0, 0].item()

                    # If score above threshold, add detection
                    if bird_score > self.threshold:
                        detections.append([x, y, window_size, window_size, bird_score])

        # Apply non-maximum suppression to remove overlapping detections
        if len(detections) > 0:
            detections = self._nms(np.array(detections), iou_threshold=0.5)

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def _nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression to remove overlapping boxes"""
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


if __name__ == "__main__":
    print("CLIP + SORT tracker ready!")
    print("Use: python run_baseline.py --config configs/clip_sort.yaml")
