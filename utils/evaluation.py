"""
Reusable evaluation utilities for all baselines
"""

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.metrics import evaluate_tracking, format_results


def convert_predictions_to_evaluation_format(predictions):
    """
    Convert predictions to format needed for metrics
    From: {frame_id: [(track_id, bbox, score), ...]}
    To: {frame_id: [(track_id, bbox), ...]}
    """
    eval_format = {}
    for frame_id, preds in predictions.items():
        eval_format[frame_id] = [(track_id, bbox) for track_id, bbox, score in preds]
    return eval_format


def convert_gt_to_evaluation_format(dataset, video_id):
    """
    Convert ground truth to evaluation format
    Returns: {frame_id: [(track_id, bbox), ...]}
    """
    gt_data = {}
    for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_id):
        gt_data[frame_id] = []
        for i, track_id in enumerate(gt_ids):
            gt_data[frame_id].append((int(track_id), gt_boxes[i].tolist()))
    return gt_data


def save_predictions_mot_format(predictions, output_file):
    """
    Save predictions in MOT Challenge format
    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for frame_id in sorted(predictions.keys()):
            for track_id, bbox, score in predictions[frame_id]:
                x, y, w, h = bbox
                f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")


def save_metrics_json(metrics, stats, output_file, model_name, video_id):
    """Save metrics and stats to JSON"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = {"model": model_name, "video_id": video_id, "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()}, "stats": stats}

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)


def save_summary_json(all_metrics, output_file, model_name, num_videos):
    """Save summary of all videos"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float, np.number)):
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    summary = {"model": model_name, "num_videos": num_videos, "average_metrics": {k: float(v) for k, v in avg_metrics.items()}, "per_video_metrics": [{k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in m.items()} for m in all_metrics]}

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    return avg_metrics


def evaluate_and_save_video(dataset, video_id, video_name, predictions, stats, output_dir, model_name, video_idx, total_videos, visualize=False):
    """
    Complete evaluation and saving for a single video

    Args:
        dataset: SMOT4SBDataset instance
        video_id: video ID
        video_name: video name
        predictions: dict {frame_id: [(track_id, bbox, score), ...]}
        stats: dict with runtime statistics
        output_dir: base output directory
        model_name: model name for saving
        video_idx: current video index (for printing)
        total_videos: total number of videos (for printing)
        visualize: whether to create visualization video (default: False)

    Returns:
        metrics: dict of evaluation metrics
    """
    print(f"\nProcessing video {video_idx}/{total_videos}: {video_name} (ID: {video_id})")
    print(f"  Frames: {stats['total_frames']}")
    print(f"  FPS: {stats['avg_fps']:.2f}")
    print(f"  Avg frame time: {stats['avg_frame_time']*1000:.1f}ms")

    # Evaluate
    pred_eval = convert_predictions_to_evaluation_format(predictions)
    gt_eval = convert_gt_to_evaluation_format(dataset, video_id)

    metrics = evaluate_tracking(gt_eval, pred_eval)
    format_results(metrics)

    # Save results
    video_output_dir = Path(output_dir) / model_name / f"video_{video_id}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions in MOT format
    pred_file = video_output_dir / f"video_{video_id}_predictions.txt"
    save_predictions_mot_format(predictions, pred_file)

    # Save metrics
    metrics_file = video_output_dir / f"video_{video_id}_metrics.json"
    save_metrics_json(metrics, stats, metrics_file, model_name, video_id)

    # Create visualization if requested
    if visualize:
        from utils.visualization import visualize_tracking_results

        vis_path = video_output_dir / "visualization.mp4"
        print(f"Creating visualization video...")
        visualize_tracking_results(dataset, video_id, predictions, vis_path)

    print(f"Saved results to {video_output_dir}")

    return metrics


def run_tracker_on_dataset(dataset, tracker, output_dir, model_name, max_videos=None, visualize=False):
    """
    Generic function to run any tracker on the dataset

    Args:
        dataset: SMOT4SBDataset instance
        tracker: tracker object with track_video(dataset, video_id) method
        output_dir: where to save results
        model_name: model/tracker name
        max_videos: max videos to process (None = all)
        visualize: whether to create visualization videos (default: False)

    Returns:
        all_metrics: list of metrics dicts for each video
        avg_metrics: dict of averaged metrics
    """
    # Get video IDs
    video_ids = dataset.get_video_ids()
    if max_videos:
        video_ids = video_ids[:max_videos]

    all_metrics = []

    # Process each video with progress bar
    print(f"\nProcessing {len(video_ids)} videos...")
    for i, video_id in enumerate(tqdm(video_ids, desc="Overall Progress", unit="video"), 1):
        video_name = dataset.videos[video_id]["name"]

        # Track video
        predictions, stats = tracker.track_video(dataset, video_id)

        # Evaluate and save
        metrics = evaluate_and_save_video(dataset, video_id, video_name, predictions, stats, output_dir, model_name, i, len(video_ids), visualize=visualize)

        all_metrics.append(metrics)

    # Compute and save summary
    print(f"\n{'='*60}")
    print(f"AVERAGE METRICS ACROSS {len(video_ids)} VIDEOS")
    print(f"{'='*60}")

    summary_file = Path(output_dir) / model_name / "summary.json"
    avg_metrics = save_summary_json(all_metrics, summary_file, model_name, len(video_ids))
    format_results(avg_metrics)

    print(f"\nEvaluation complete! Results saved to {output_dir}/{model_name}")

    return all_metrics, avg_metrics


# Example usage
if __name__ == "__main__":
    from utils.data_loader import SMOT4SBDataset

    # Test conversion functions
    dummy_predictions = {1: [(1, [100, 100, 20, 20], 0.9), (2, [200, 200, 15, 15], 0.85)], 2: [(1, [105, 105, 20, 20], 0.92), (2, [205, 205, 15, 15], 0.87)]}

    eval_format = convert_predictions_to_evaluation_format(dummy_predictions)
    print("Predictions converted to evaluation format:")
    print(eval_format)

    # Test saving MOT format
    save_predictions_mot_format(dummy_predictions, "test_predictions.txt")
    print("\nSaved test predictions to test_predictions.txt")

    print("\nEvaluation utils ready!")
