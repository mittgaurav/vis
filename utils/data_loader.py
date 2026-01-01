"""
Data loader for SMOT4SB dataset (COCO format with tracking)
"""

import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
np.bool = bool


class SMOT4SBDataset:
    """Load and parse SMOT4SB dataset"""

    def __init__(self, data_path="data/phase_1/train", annotation_file="data/annotations/train.json"):
        """
        Args:
            data_path: Path to data folder containing phase_1, phase_2
            annotation_file: Path to train.json
        """
        self.data_path = Path(data_path)
        self.annotation_file = annotation_file

        # Load annotations
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        # Parse data structures
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.videos = {vid["id"]: vid for vid in self.coco_data["videos"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Group annotations by image_id
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)

        # Group images by video_id (scene)
        self.images_by_video = defaultdict(list)
        for img in self.coco_data["images"]:
            self.images_by_video[img["video_id"]].append(img)

        # Sort frames within each video by frame_id
        for vid_id in self.images_by_video:
            self.images_by_video[vid_id].sort(key=lambda x: x["frame_id"])

        print(f"Loaded dataset:")
        print(f"  Videos (folders): {len(self.videos)}")
        print(f"  Total frames (images): {len(self.images)}")
        print(f"  Total annotations: {len(self.coco_data['annotations'])}")

    def get_video_ids(self):
        """Get all video (scene) IDs"""
        return sorted(self.videos.keys())

    def get_video_frames(self, video_id):
        """Get all frame info for a video (sorted by frame_id)"""
        return self.images_by_video[video_id]

    def get_frame_annotations(self, image_id):
        """
        Get ground truth annotations for a frame
        Returns list of dicts with keys: track_id, bbox, conf, category_id
        """
        return self.annotations_by_image[image_id]

    def load_frame_image(self, image_info):
        """Load image array from disk"""
        # Handle both phase_1 and phase_2
        img_path = self.data_path / image_info["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        return img

    def get_ground_truth_tracks(self, video_id):
        """
        Get all ground truth tracks for a video
        Returns dict: {track_id: [(frame_id, bbox), ...]}
        """
        frames = self.get_video_frames(video_id)
        tracks = defaultdict(list)

        for frame_info in frames:
            frame_id = frame_info["frame_id"]
            annotations = self.get_frame_annotations(frame_info["id"])

            for ann in annotations:
                track_id = ann["track_id"]
                bbox = ann["bbox"]  # [x, y, w, h]
                tracks[track_id].append((frame_id, bbox))

        return tracks

    def iterate_video(self, video_id):
        """
        Iterator for a video sequence
        Yields: (frame_id, image, gt_boxes, gt_track_ids)
        """
        frames = self.get_video_frames(video_id)

        for frame_info in frames:
            # Load image
            image = self.load_frame_image(frame_info)

            # Get annotations
            annotations = self.get_frame_annotations(frame_info["id"])

            gt_boxes = []
            gt_track_ids = []

            for ann in annotations:
                gt_boxes.append(ann["bbox"])  # [x, y, w, h]
                gt_track_ids.append(ann["track_id"])

            yield frame_info["frame_id"], image, np.array(gt_boxes), np.array(gt_track_ids)

    def iterate_all_videos(self):
        """
        Iterator over all videos
        Yields: (video_id, video_name, frame_iterator)
        """
        for video_id in self.get_video_ids():
            video_name = self.videos[video_id]["name"]
            yield video_id, video_name, self.iterate_video(video_id)


def convert_bbox_to_mot_format(bbox):
    """
    Convert COCO bbox [x, y, w, h] to MOT format [x, y, w, h]
    (Already same format, but keeping function for consistency)
    """
    return bbox


def convert_bbox_to_xyxy(bbox):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def convert_bbox_center(bbox):
    """Get center point of bbox [x, y, w, h]"""
    x, y, w, h = bbox
    return [x + w / 2, y + h / 2]


# Example usage
if __name__ == "__main__":
    # Test data loader
    dataset = SMOT4SBDataset(data_path="data/phase_1/train", annotation_file="data/annotations/train.json")

    # Test loading one video
    video_ids = dataset.get_video_ids()
    print(f"\nFirst video ID: {video_ids[0]}")

    first_video_id = video_ids[0]
    frames = dataset.get_video_frames(first_video_id)
    print(f"Number of frames: {len(frames)}")

    # Test iteration
    print("\nTesting frame iteration (first 3 frames):")
    for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(first_video_id):
        print(f"Frame {frame_id}: shape={image.shape}, birds={len(gt_boxes)}, track_ids={gt_ids}")
        break

    print("\nData loader ready!")
