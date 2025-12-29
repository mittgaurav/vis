"""
Complete YOLO + SORT baseline for bird tracking
Supports YOLO12n, YOLO12s, YOLO12m, YOLO12x variants
"""

import argparse
import time
import numpy as np
from tqdm import tqdm

from trackers.sort_tracker import Sort
from utils.data_loader import SMOT4SBDataset
from utils.evaluation import run_tracker_on_dataset


class YOLODetector:
    """Wrapper for YOLO detection"""

    def __init__(self, model_name="yolo12n", conf_threshold=0.1, device="cpu", detect_all_classes=False):
        """
        Args:
            model_name: 'yolo12n', 'yolo12s', 'yolo12m', 'yolo12x' or 'yolo11n', 'yolo11s', etc.
            conf_threshold: confidence threshold for detections (lower for small birds)
            device: 'cpu' or 'cuda'
            detect_all_classes: if True, detect all objects (for debugging)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.detect_all_classes = detect_all_classes

        # Load YOLO model
        try:
            from ultralytics import YOLO

            print(f"Loading {model_name}...")
            self.model = YOLO(f"{model_name}.pt")
            self.model.to(device)
            print(f"Model loaded on {device}")
            print(f"Detection config: conf_threshold={conf_threshold}, detect_all_classes={detect_all_classes}")
        except ImportError:
            print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise

    def detect(self, image, debug=False):
        """
        Run detection on image

        Args:
            image: BGR image (H, W, 3)
            debug: if True, print detection info

        Returns:
            detections: numpy array (N, 5) of [x, y, w, h, confidence]
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)

        detections = []
        all_detections_info = []

        # Extract bird detections (class 14 is bird in COCO)
        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Get box in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert to xywh
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                # Store info for debugging
                all_detections_info.append((cls, conf, w * h))

                # Filter for birds (class 14) or all classes
                if self.detect_all_classes or cls == 14:
                    detections.append([x, y, w, h, conf])

        # Debug output
        if debug and len(all_detections_info) > 0:
            classes_detected = set([cls for cls, _, _ in all_detections_info])
            print(f"  YOLO found {len(all_detections_info)} objects, classes: {classes_detected}")
            print(f"  Birds (class 14): {sum(1 for cls, _, _ in all_detections_info if cls == 14)}")
            if len(all_detections_info) > 0:
                areas = [area for _, _, area in all_detections_info]
                print(f"  Detection sizes: min={min(areas):.1f}, max={max(areas):.1f}, mean={np.mean(areas):.1f} pixels²")

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))


class YOLOSORTTracker:
    """Complete YOLO + SORT tracking pipeline"""

    def __init__(self, model_name="yolo12n", conf_threshold=0.1, max_age=1, min_hits=3, iou_threshold=0.3, device="cpu", detect_all_classes=False):
        """
        Args:
            model_name: YOLO model name (yolo12n, yolo12s, yolo12m, yolo11n, etc.)
            conf_threshold: detection confidence threshold (lower for small objects)
            max_age: SORT max_age parameter
            min_hits: SORT min_hits parameter
            iou_threshold: SORT IoU threshold for matching
            device: 'cpu' or 'cuda'
            detect_all_classes: if True, detect all objects not just birds
        """
        self.detector = YOLODetector(model_name, conf_threshold, device, detect_all_classes)
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.device = device

    def track_video(self, dataset, video_id):
        """
        Track all birds in a video

        Returns:
            predictions: dict {frame_id: [(track_id, bbox, score), ...]}
            stats: dict with runtime statistics
        """
        self.tracker.reset()
        predictions = {}

        frame_times = []
        total_detections = 0

        # Get total frames for progress bar
        frames = list(dataset.iterate_video(video_id))
        video_name = dataset.videos[video_id]["name"]

        # Debug first frame
        print(f"\n  Debugging first frame of video {video_name}...")
        if len(frames) > 0:
            first_frame_id, first_image, first_gt_boxes, first_gt_ids = frames[0]
            print(f"  Ground truth: {len(first_gt_boxes)} birds")
            if len(first_gt_boxes) > 0:
                gt_areas = [w * h for x, y, w, h in first_gt_boxes]
                print(f"  GT bird sizes: min={min(gt_areas):.1f}, max={max(gt_areas):.1f}, mean={np.mean(gt_areas):.1f} pixels²")
            _ = self.detector.detect(first_image, debug=True)

        for frame_id, image, gt_boxes, gt_ids in tqdm(frames, desc=f"Video {video_name}", unit="frame", leave=False):
            start_time = time.time()

            # Detect birds
            detections = self.detector.detect(image)
            total_detections += len(detections)

            # Track
            tracks = self.tracker.update(detections)

            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            # Store predictions
            predictions[frame_id] = []
            for track in tracks:
                x, y, w, h, track_id = track
                # Find original detection score
                score = 1.0  # Default if not found
                for det in detections:
                    if abs(det[0] - x) < 1 and abs(det[1] - y) < 1:
                        score = det[4]
                        break

                predictions[frame_id].append((int(track_id), [x, y, w, h], score))

        stats = {"total_frames": len(frame_times), "total_time": sum(frame_times), "avg_fps": 1.0 / (sum(frame_times) / len(frame_times)) if len(frame_times) > 0 else 0, "avg_frame_time": np.mean(frame_times) if len(frame_times) > 0 else 0, "total_detections": total_detections, "avg_detections_per_frame": total_detections / len(frame_times) if len(frame_times) > 0 else 0}

        print(f"  Total detections across video: {total_detections}")
        print(f"  Avg detections per frame: {stats['avg_detections_per_frame']:.2f}")

        return predictions, stats


def run_baseline(data_root, annotation_file, output_dir, model_name="yolo12n", max_videos=None, visualize=False, conf_threshold=0.1, detect_all_classes=False):
    """
    Run complete YOLO + SORT baseline

    Args:
        data_root: path to data folder
        annotation_file: path to train.json
        output_dir: where to save results
        model_name: YOLO model name (yolo12n, yolo12s, yolo12m, yolo11n, etc.)
        max_videos: maximum videos to process (None = all)
        visualize: whether to create visualization videos (default: False)
        conf_threshold: detection confidence threshold (lower for small birds)
        detect_all_classes: detect all objects, not just birds (for debugging)
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name} + SORT baseline")
    print(f"Config: conf_threshold={conf_threshold}, detect_all={detect_all_classes}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = SMOT4SBDataset(data_root, annotation_file)

    # Initialize tracker
    tracker = YOLOSORTTracker(model_name=model_name, conf_threshold=conf_threshold, max_age=1, min_hits=3, iou_threshold=0.3, device="cpu", detect_all_classes=detect_all_classes)

    # Run evaluation using generic utility
    all_metrics, avg_metrics = run_tracker_on_dataset(dataset=dataset, tracker=tracker, output_dir=output_dir, model_name=f"{model_name}_conf{conf_threshold}", max_videos=max_videos, visualize=visualize)

    return all_metrics, avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO + SORT baseline")
    parser.add_argument("--data_root", type=str, default="data/phase_1/train", help="Path to data folder")
    parser.add_argument("--annotation_file", type=str, default="data/annotations/train.json")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--model", type=str, default="yolo12n", choices=["yolo12n", "yolo12s", "yolo12m", "yolo12x", "yolo11n", "yolo11s", "yolo11m", "yolo11x"], help="YOLO model variant")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
    parser.add_argument("--visualize", action="store_true", help="Create visualization videos (WARNING: may be slow/heavy)")
    parser.add_argument("--conf_threshold", type=float, default=0.1, help="Detection confidence threshold (try 0.05 for small birds)")
    parser.add_argument("--detect_all_classes", action="store_true", help="Detect all objects, not just birds (debugging)")

    args = parser.parse_args()

    all_metrics, avg_metrics = run_baseline(data_root=args.data_root, annotation_file=args.annotation_file, output_dir=args.output_dir, model_name=args.model, max_videos=args.max_videos, visualize=args.visualize, conf_threshold=args.conf_threshold, detect_all_classes=args.detect_all_classes)
