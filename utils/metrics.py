"""
Tracking metrics: Precision, Recall, IoU, MOTA, HOTA, DotD
Specifically tuned for Small Multi-Object Tracking (SMOT) for birds.
"""

import numpy as np
import motmetrics as mm
from utils.hota_trackeval import compute_hota_trackeval


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


def compute_detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.1):
    """
    Compute Precision, Recall, and mean IoU for detection
    Default threshold set to 0.1 for small targets.
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

    # Sort GT boxes by some criteria if needed, here we just iterate
    for i in range(len(gt_boxes)):
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


def compute_tracking_metrics(gt_data, pred_data, iou_threshold=0.1):
    """
    Compute MOTA, MOTP, etc. using motmetrics with a relaxed threshold for SMOT.
    """
    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame IDs
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for frame_id in all_frames:
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])

        if not gt_frame and not pred_frame:
            continue

        gt_ids = [item[0] for item in gt_frame]
        gt_boxes = np.array([item[1] for item in gt_frame]) if gt_frame else np.empty((0, 4))
        pred_ids = [item[0] for item in pred_frame]
        pred_boxes = np.array([item[1] for item in pred_frame]) if pred_frame else np.empty((0, 4))

        # Distance matrix: 1 - IoU
        # If IoU is below threshold, distance should be > 1 to count as a miss in motmetrics
        dist_matrix = np.full((len(gt_boxes), len(pred_boxes)), np.nan)
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou = compute_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    dist_matrix[i, j] = 1.0 - iou

        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "motp", "num_switches", "num_false_positives", "num_misses", "precision", "recall"],
        name="tracking",
    )
    return summary.to_dict("records")[0] if len(summary) > 0 else {}


def compute_dotd(gt_data, pred_data):
    """
    Compute Dot Distance (DotD) - Average center distance.
    This is often more reliable than IoU for small targets.
    """
    distances = []
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for frame_id in all_frames:
        gt_dict = {tid: bbox for tid, bbox in gt_data.get(frame_id, [])}
        pred_dict = {tid: bbox for tid, bbox in pred_data.get(frame_id, [])}

        for gt_id, gt_bbox in gt_dict.items():
            if gt_id in pred_dict:
                pred_bbox = pred_dict[gt_id]
                gt_c = [gt_bbox[0] + gt_bbox[2]/2, gt_bbox[1] + gt_bbox[3]/2]
                pred_c = [pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2]
                dist = np.sqrt((gt_c[0] - pred_c[0])**2 + (gt_c[1] - pred_c[1])**2)
                distances.append(dist)

    return np.mean(distances) if distances else float("inf")


def evaluate_tracking(gt_data, pred_data, iou_threshold=0.1):
    """
    Main evaluation entry point with relaxed defaults for small birds.
    """
    all_gt_boxes = []
    all_pred_boxes = []

    for frame_id in gt_data:
        all_gt_boxes.extend([bbox for _, bbox in gt_data[frame_id]])
        if frame_id in pred_data:
            all_pred_boxes.extend([bbox for _, bbox in pred_data[frame_id]])

    precision, recall, mean_iou = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold)
    tracking_metrics = compute_tracking_metrics(gt_data, pred_data, iou_threshold)
    dotd = compute_dotd(gt_data, pred_data)

    hota = compute_hota_trackeval(gt_data, pred_data)


    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "dotd": dotd,
        "hota": hota,
        **tracking_metrics,
    }

def format_results(m):
    print("\n" + "="*50 + "\nTRACKING RESULTS (SMOT Tuned)\n" + "="*50)
    print(f"Precision: {m.get('precision',0):.4f} | Recall: {m.get('recall',0):.4f}")
    print(f"MOTA: {m.get('mota',0):.4f} | HOTA (approx): {m.get('hota',0):.4f}")
    print(f"ID Sw: {m.get('num_switches',0):.0f} | DotD: {m.get('dotd',0):.2f} px")
    print("="*50 + "\n")


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
