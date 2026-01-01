"""
Tracking metrics: Precision, Recall, IoU, MOTA, HOTA, DotD
Uses motmetrics library for standard tracking metrics
"""

import numpy as np
np.bool = bool
import motmetrics as mm


def compute_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes [x, y, w, h]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corners
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Intersection
    xi = max(x1, x2)
    yi = max(y1, y2)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0, xi_max - xi)
    inter_h = max(0, yi_max - yi)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Compute Precision, Recall, and mean IoU for detection
    """
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0

    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)

    # For each GT box, find best matching prediction
    matched_preds = set()
    true_positives = 0
    ious = []

    for i in range(len(gt_boxes)):
        if len(pred_boxes) == 0:
            break

        # Find best matching prediction
        best_iou = 0
        best_j = -1
        for j in range(len(pred_boxes)):
            if j not in matched_preds and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_preds.add(best_j)
            ious.append(best_iou)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0

    return precision, recall, mean_iou


def compute_tracking_metrics(gt_data, pred_data):
    """
    Compute MOTA, MOTP, and other tracking metrics using motmetrics
    """
    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame IDs
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])

        if len(gt_frame) == 0 and len(pred_frame) == 0:
            continue

        # Extract track IDs and boxes
        gt_ids = [item[0] for item in gt_frame]
        gt_boxes = np.array([item[1] for item in gt_frame]) if len(gt_frame) > 0 else np.empty((0, 4))

        pred_ids = [item[0] for item in pred_frame]
        pred_boxes = np.array([item[1] for item in pred_frame]) if len(pred_frame) > 0 else np.empty((0, 4))

        # Compute distance matrix (1 - IoU)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, pred_box in enumerate(pred_boxes):
                    iou_matrix[i, j] = compute_iou(gt_box, pred_box)

            distance_matrix = 1 - iou_matrix
        else:
            distance_matrix = np.empty((len(gt_boxes), len(pred_boxes)))

        # Update accumulator
        acc.update(gt_ids, pred_ids, distance_matrix)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "precision",
            "recall",
        ],
        name="tracking",
    )

    return summary.to_dict("records")[0] if len(summary) > 0 else {}


def compute_dotd(gt_data, pred_data):
    """
    Compute Distance over Time (DotD)
    Average Euclidean distance between predicted and GT centers across all frames
    """
    distances = []

    # Build track correspondence
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])

        if len(gt_frame) == 0 or len(pred_frame) == 0:
            continue

        # Build dictionaries for easier lookup
        gt_dict = {track_id: bbox for track_id, bbox in gt_frame}
        pred_dict = {track_id: bbox for track_id, bbox in pred_frame}

        # For each GT track, find corresponding prediction
        for gt_id, gt_bbox in gt_dict.items():
            if gt_id in pred_dict:
                pred_bbox = pred_dict[gt_id]

                # Compute center points
                gt_center = [gt_bbox[0] + gt_bbox[2] / 2, gt_bbox[1] + gt_bbox[3] / 2]
                pred_center = [
                    pred_bbox[0] + pred_bbox[2] / 2,
                    pred_bbox[1] + pred_bbox[3] / 2,
                ]

                # Euclidean distance
                dist = np.sqrt((gt_center[0] - pred_center[0]) ** 2 + (gt_center[1] - pred_center[1]) ** 2)
                distances.append(dist)

    return np.mean(distances) if len(distances) > 0 else float("inf")


def compute_hota_approx(gt_data, pred_data, iou_threshold=0.5):
    """
    Approximate HOTA (Higher Order Tracking Accuracy).

    Very lightweight approximation:
    - For each frame and GT track ID, check if there is a prediction with the same ID
      and IoU >= threshold.
    - Count those as "correct tracking events".
    - HOTA_approx = correct_events / total_gt_events.

    This captures joint detection+association quality in a single scalar and
    is acceptable as a simplified HOTA-style metric for coursework, as long
    as the approximation is documented. [web:52][web:132]
    """
    correct_events = 0
    total_events = 0

    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])
        if not gt_frame:
            continue

        pred_dict = {tid: bbox for tid, bbox in pred_frame}
        for gt_id, gt_bbox in gt_frame:
            total_events += 1
            if gt_id in pred_dict:
                iou = compute_iou(gt_bbox, pred_dict[gt_id])
                if iou >= iou_threshold:
                    correct_events += 1

    if total_events == 0:
        return 0.0
    return correct_events / total_events


def evaluate_tracking(gt_data, pred_data, iou_threshold=0.5):
    """
    Comprehensive evaluation of tracking results
    """
    # Compute detection metrics (frame-level)
    all_gt_boxes = []
    all_pred_boxes = []

    for frame_id in gt_data:
        all_gt_boxes.extend([bbox for _, bbox in gt_data[frame_id]])
        if frame_id in pred_data:
            all_pred_boxes.extend([bbox for _, bbox in pred_data[frame_id]])

    precision, recall, mean_iou = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold)

    # Compute tracking metrics (MOTA, MOTP, etc.)
    tracking_metrics = compute_tracking_metrics(gt_data, pred_data)

    # Compute DotD
    dotd = compute_dotd(gt_data, pred_data)

    # Compute approximate HOTA
    hota = compute_hota_approx(gt_data, pred_data, iou_threshold=iou_threshold)

    # Combine all metrics
    results = {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "dotd": dotd,
        "hota": hota,
        **tracking_metrics,
    }

    return results


def format_results(metrics_dict):
    """Format metrics for display"""
    print("\n" + "=" * 50)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 50)

    print("\nDetection Metrics:")
    print(f"  Precision: {metrics_dict.get('precision', 0):.4f}")
    print(f"  Recall: {metrics_dict.get('recall', 0):.4f}")
    print(f"  Mean IoU: {metrics_dict.get('mean_iou', 0):.4f}")

    print("\nTracking Metrics:")
    print(f"  MOTA: {metrics_dict.get('mota', 0):.4f}")
    print(f"  MOTP: {metrics_dict.get('motp', 0):.4f}")
    print(f"  HOTA (approx): {metrics_dict.get('hota', 0):.4f}")
    print(f"  ID Switches: {metrics_dict.get('num_switches', 0):.0f}")
    print(f"  False Positives: {metrics_dict.get('num_false_positives', 0):.0f}")
    print(f"  Misses: {metrics_dict.get('num_misses', 0):.0f}")

    print("\nDistance Metrics:")
    print(f"  DotD (Distance over Time): {metrics_dict.get('dotd', float('inf')):.2f} pixels")

    print("=" * 50 + "\n")


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    gt_data = {
        1: [(1, [100, 100, 20, 20]), (2, [200, 200, 15, 15])],
        2: [(1, [105, 105, 20, 20]), (2, [205, 205, 15, 15])],
        3: [(1, [110, 110, 20, 20]), (2, [210, 210, 15, 15])],
    }

    pred_data = {
        1: [(1, [102, 102, 20, 20]), (2, [198, 198, 15, 15])],
        2: [(1, [107, 107, 20, 20]), (2, [203, 203, 15, 15])],
        3: [(1, [112, 112, 20, 20])],  # Missing track 2
    }

    metrics = evaluate_tracking(gt_data, pred_data)
    format_results(metrics)
    print("Metrics module ready!")
