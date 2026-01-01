"""
DINO features + appearance-based tracking
Uses DINOv2 for feature extraction and similarity-based matching
"""
import numpy as np
import torch
import cv2
from baselines.base_tracker import BaseTracker
from trackers.sort_tracker import Sort


class DINOSORTTracker(BaseTracker):
    """
    DINOv2 feature-based tracker
    Uses background subtraction for detection + DINO features for matching
    """

    def _initialize_detector(self):
        """Initialize DINOv2 model and background subtractor"""
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed")

        detector_config = self.config['detector']
        model_name = detector_config.get('model_name', 'dinov2_vits14')

        print(f"Loading DINOv2 {model_name}...")

        # Load DINOv2 from torch hub
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        print(f"DINOv2 loaded on {self.device}")

        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

        # Store params
        self.min_area = detector_config['params'].get('min_area', 50)
        self.max_area = detector_config['params'].get('max_area', 10000)
        self.conf_threshold = detector_config.get('conf_threshold', 0.5)

        print(f"Background subtraction config: min_area={self.min_area}, max_area={self.max_area}")

        return self.dino_model

    def _initialize_tracker(self):
        """Initialize SORT tracker with appearance features"""
        tracker_params = self.config['tracker']['params']

        # Use standard SORT but we'll enhance with DINO features
        tracker = Sort(
            max_age=tracker_params.get('max_age', 3),  # Longer for appearance-based
            min_hits=tracker_params.get('min_hits', 3),
            iou_threshold=tracker_params.get('iou_threshold', 0.3)
        )

        return tracker

    def _detect_frame(self, image):
        """
        Detect birds using background subtraction

        Args:
            image: BGR image (H, W, 3)

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on area (normalized)
                conf = min(1.0, area / 1000.0)
                detections.append([x, y, w, h, conf])
                if conf >= self.conf_threshold:
                    detections.append([x, y, w, h, conf])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def extract_dino_features(self, image, bbox):
        """
        Extract DINO features for a bounding box

        Args:
            image: BGR image
            bbox: [x, y, w, h]

        Returns:
            features: numpy array of DINO features
        """
        x, y, w, h = [int(v) for v in bbox]

        # Crop region
        crop = image[y:y + h, x:x + w]

        if crop.size == 0:
            return np.zeros(384)  # Default feature size for dinov2_vits14

        # Convert to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to 224x224 for DINO
        crop_resized = cv2.resize(crop_rgb, (224, 224))

        # Convert to tensor
        crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.detector(crop_tensor)
            features = features.cpu().numpy().flatten()

        return features


if __name__ == "__main__":
    print("DINO + SORT tracker ready!")
    print("Use: python run_baseline.py --config configs/dino_sort.yaml")
