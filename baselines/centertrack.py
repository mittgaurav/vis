"""
CenterTrack - End-to-end detection and tracking
Uses the official CenterTrack implementation in the parent directory
"""
import numpy as np
import time
from tqdm import tqdm
from baselines.base_tracker import BaseTracker

# Import actual CenterTrack detector
try:
    import sys
    sys.path.append("..")  # parent dir
    from centertrack.detector import Detector
except ImportError:
    Detector = None
    print("WARNING: CenterTrack not found. Install from https://github.com/xingyizhou/CenterTrack")


class CenterTracker(BaseTracker):
    """
    CenterTrack end-to-end tracker
    """

    def _initialize_detector(self):
        """Initialize CenterTrack model"""
        if Detector is None:
            print("CenterTrack detector not available, will run placeholder")
            return None

        # Use config options if needed (can pass via self.config)
        opt = {}  # add model path or params if necessary
        detector = Detector(opt)
        return detector

    def _initialize_tracker(self):
        """CenterTrack does detection + tracking together"""
        # no separate tracker needed
        return None

    def _detect_frame(self, image):
        """
        CenterTrack processes frames and outputs tracks directly
        """
        if self.detector is None:
            return np.empty((0, 5))

        # CenterTrack expects previous frame? handled in track_video
        results = self.detector.run(image)
        detections = []

        for track in results['results']:
            # track format: [x, y, w, h, track_id, score]
            x, y, w, h, track_id, score = track
            detections.append([x, y, w, h, score])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    def track_video(self, dataset, video_id):
        """
        CenterTrack end-to-end tracking
        """
        predictions = {}
        frame_times = []

        frames = list(dataset.iterate_video(video_id))
        video_name = dataset.videos[video_id]['name']

        print(f"\nProcessing video {video_name}...")

        prev_image = None

        for frame_id, image, gt_boxes, gt_ids in tqdm(
                frames, desc=f"Video {video_name}", unit="frame", leave=False
        ):
            start_time = time.time()

            if self.detector is not None:
                results = self.detector.run(image, prev_image)
                tracks = results['results']
            else:
                tracks = []

            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            predictions[frame_id] = []
            for track in tracks:
                # track format: [x, y, w, h, track_id, score]
                x, y, w, h, track_id, score = track
                predictions[frame_id].append((int(track_id), [x, y, w, h], score))

            prev_image = image.copy()

        stats = {
            'total_frames': len(frame_times),
            'total_time': sum(frame_times),
            'avg_fps': 1.0 / (sum(frame_times) / len(frame_times)) if len(frame_times) > 0 else 0,
            'avg_frame_time': np.mean(frame_times) if len(frame_times) > 0 else 0,
            'total_detections': sum(len(p) for p in predictions.values()),
            'avg_detections_per_frame': np.mean([len(p) for p in predictions.values()]) if len(predictions) > 0 else 0
        }

        return predictions, stats


if __name__ == "__main__":
    print("CenterTrack tracker ready!")
    print("Ensure the official CenterTrack repo is in the parent directory")
