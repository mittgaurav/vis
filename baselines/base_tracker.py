"""
Abstract base class for all trackers
All baseline implementations should inherit from this
"""
from abc import ABC, abstractmethod
import time
import numpy as np
np.bool = bool
from tqdm import tqdm


class BaseTracker(ABC):
    """
    Abstract base class for tracking pipelines

    Subclasses must implement:
    - _initialize_detector()
    - _initialize_tracker()
    - _detect_frame()
    """

    def __init__(self, config):
        """
        Args:
            config: dict with configuration parameters
        """
        self.config = config
        self.name = config.get('name', 'unnamed_tracker')
        self.device = config.get('device', 'cpu')

        # Initialize detector and tracker
        self.detector = self._initialize_detector()
        self.tracker = self._initialize_tracker()

        # Stats tracking
        self.frame_count = 0
        self.total_detections = 0

    @abstractmethod
    def _initialize_detector(self):
        """Initialize and return detector (e.g., YOLO model)"""
        pass

    @abstractmethod
    def _initialize_tracker(self):
        """Initialize and return tracker (e.g., SORT)"""
        pass

    @abstractmethod
    def _detect_frame(self, image):
        """
        Run detection on a single frame

        Args:
            image: numpy array (H, W, 3)

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        pass

    def _track_frame(self, detections):
        """
        Run tracking on detections

        Args:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]

        Returns:
            tracks: numpy array (M, 5) of [x, y, w, h, track_id]
        """
        return self.tracker.update(detections)

    def track_video(self, dataset, video_id):
        """
        Track all objects in a video

        Args:
            dataset: SMOT4SBDataset instance
            video_id: video ID to process

        Returns:
            predictions: dict {frame_id: [(track_id, bbox, score), ...]}
            stats: dict with runtime statistics
        """
        # Reset tracker for new video
        self.tracker.reset()
        predictions = {}
        frame_times = []
        total_detections = 0

        # Get frames
        frames = list(dataset.iterate_video(video_id))
        video_name = dataset.videos[video_id]['name']

        # Debug first frame if verbose
        if self.config.get('debug', {}).get('verbose', False):
            self._debug_first_frame(frames)

        # Process each frame
        for frame_id, image, gt_boxes, gt_ids in tqdm(
                frames,
                desc=f"Video {video_name}",
                unit="frame",
                leave=False,
                disable=not self.config.get('debug', {}).get('verbose', True)
        ):
            start_time = time.time()

            # Detect
            detections = self._detect_frame(image)
            total_detections += len(detections)

            # Track
            tracks = self._track_frame(detections)

            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            # Store predictions with scores
            predictions[frame_id] = []
            for track in tracks:
                x, y, w, h, track_id = track

                # Find corresponding detection score
                score = self._find_detection_score(detections, x, y)
                predictions[frame_id].append((int(track_id), [x, y, w, h], score))

        # Compute statistics
        stats = {
            'total_frames': len(frame_times),
            'total_time': sum(frame_times),
            'avg_fps': 1.0 / (sum(frame_times) / len(frame_times)) if len(frame_times) > 0 else 0,
            'avg_frame_time': np.mean(frame_times) if len(frame_times) > 0 else 0,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(frame_times) if len(frame_times) > 0 else 0
        }

        if self.config.get('debug', {}).get('verbose', False):
            print(f"  Total detections: {total_detections}")
            print(f"  Avg detections/frame: {stats['avg_detections_per_frame']:.2f}")

        return predictions, stats

    def _find_detection_score(self, detections, x, y):
        """Find score for a tracked object from original detections"""
        score = 1.0
        for det in detections:
            if abs(det[0] - x) < 1 and abs(det[1] - y) < 1:
                score = det[4]
                break
        return score

    def _debug_first_frame(self, frames):
        """Debug output for first frame"""
        if len(frames) == 0:
            return

        frame_id, image, gt_boxes, gt_ids = frames[0]
        print(f"\n  Debugging first frame...")
        print(f"  Ground truth: {len(gt_boxes)} objects")

        if len(gt_boxes) > 0:
            gt_areas = [w * h for x, y, w, h in gt_boxes]
            print(f"  GT sizes: min={min(gt_areas):.1f}, max={max(gt_areas):.1f}, mean={np.mean(gt_areas):.1f} pixels²")

        # Run detection on first frame
        detections = self._detect_frame(image)
        print(f"  Detections: {len(detections)}")

        if len(detections) > 0:
            det_areas = [w * h for x, y, w, h, _ in detections]
            print(
                f"  Detection sizes: min={min(det_areas):.1f}, max={max(det_areas):.1f}, mean={np.mean(det_areas):.1f} pixels²")


# Example usage
if __name__ == "__main__":
    print("BaseTracker is an abstract class. Use YOLOSORTTracker or other implementations.")
