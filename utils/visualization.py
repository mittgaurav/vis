"""
Visualization utilities for bird tracking
"""

import cv2
import numpy as np
np.bool = bool
from pathlib import Path


def draw_boxes(image, boxes, track_ids=None, scores=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image

    Args:
        image: numpy array (H, W, 3)
        boxes: list of [x, y, w, h] or numpy array (N, 4)
        track_ids: list of track IDs (optional)
        scores: list of confidence scores (optional)
        color: BGR color tuple
        thickness: line thickness

    Returns:
        image with boxes drawn
    """
    img_copy = image.copy()

    if len(boxes) == 0:
        return img_copy

    boxes = np.array(boxes)

    for i, bbox in enumerate(boxes):
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        # Draw box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        label_parts = []
        if track_ids is not None:
            label_parts.append(f"ID:{track_ids[i]}")
        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            cv2.rectangle(img_copy, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), font_thickness)

    return img_copy


def visualize_tracking_results(dataset, video_id, predictions, output_path, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
    """
    Create video visualization comparing ground truth and predictions

    Args:
        dataset: SMOT4SBDataset instance
        video_id: video ID to visualize
        predictions: dict {frame_id: [(track_id, bbox, score), ...]}
        output_path: path to save video
        gt_color: color for ground truth boxes (BGR)
        pred_color: color for prediction boxes (BGR)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get first frame to determine video properties
    frames = list(dataset.iterate_video(video_id))
    if len(frames) == 0:
        print(f"No frames found for video {video_id}")
        return

    first_frame_id, first_image, _, _ = frames[0]
    height, width = first_image.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width * 2, height))

    for frame_id, image, gt_boxes, gt_ids in frames:
        # Draw ground truth on left
        img_gt = draw_boxes(image, gt_boxes, gt_ids, color=gt_color)
        cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw predictions on right
        img_pred = image.copy()
        if frame_id in predictions:
            pred_ids = [p[0] for p in predictions[frame_id]]
            pred_boxes = [p[1] for p in predictions[frame_id]]
            pred_scores = [p[2] for p in predictions[frame_id]]
            img_pred = draw_boxes(img_pred, pred_boxes, pred_ids, pred_scores, color=pred_color)

        cv2.putText(img_pred, "Predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenate side by side
        combined = np.hstack([img_gt, img_pred])
        out.write(combined)

    out.release()
    print(f"Saved visualization to {output_path}")


def save_frame_visualization(image, gt_boxes, gt_ids, pred_boxes, pred_ids, pred_scores, output_path):
    """Save single frame comparison"""
    img_gt = draw_boxes(image, gt_boxes, gt_ids, color=(0, 255, 0))
    img_pred = draw_boxes(image, pred_boxes, pred_ids, pred_scores, color=(255, 0, 0))

    combined = np.hstack([img_gt, img_pred])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def visualize_ground_truth(dataset, video_id, output_path, max_frames=None):
    """
    Visualize ground truth annotations for a video

    Args:
        dataset: SMOT4SBDataset instance
        video_id: video ID
        output_path: path to save video
        max_frames: maximum frames to process (None = all)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = list(dataset.iterate_video(video_id))
    if max_frames:
        frames = frames[:max_frames]

    if len(frames) == 0:
        return

    first_frame_id, first_image, _, _ = frames[0]
    height, width = first_image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    for frame_id, image, gt_boxes, gt_ids in frames:
        img_with_boxes = draw_boxes(image, gt_boxes, gt_ids, color=(0, 255, 0))

        # Add frame info
        info_text = f"Frame: {frame_id} | Birds: {len(gt_boxes)}"
        cv2.putText(img_with_boxes, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(img_with_boxes)

    out.release()
    print(f"Saved ground truth visualization to {output_path}")


# Example usage
if __name__ == "__main__":
    from data_loader import SMOT4SBDataset

    # Load dataset
    dataset = SMOT4SBDataset()

    # Visualize first video ground truth
    video_ids = dataset.get_video_ids()
    first_video_id = video_ids[0]

    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_ground_truth(dataset, first_video_id, output_dir / f"video_{first_video_id}_gt.mp4", max_frames=100)  # Limit for testing

    print("Visualization complete!")
