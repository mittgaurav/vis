import numpy as np
from sklearn.linear_model import LinearRegression

import json


class EnsembleWeightOptimizer:
    """
    Optimize ensemble detector weights (YOLO, MOG2, OpticalFlow)
    via linear regression on training data.
    """

    def __init__(self, iou_threshold=0.1):
        self.iou_threshold = iou_threshold
        self.model = None
        self.optimal_weights = None

    def iou_xywh(self, box1, box2):
        """Calculate IoU for [x, y, w, h] format"""
        x1a, y1a, w1, h1 = box1
        x1b, y1b, w1b, h1b = box2

        x_inter_min = max(x1a, x1b)
        y_inter_min = max(y1a, y1b)
        x_inter_max = min(x1a + w1, x1b + w1b)
        y_inter_max = min(y1a + h1, y1b + h1b)

        inter_w = max(0, x_inter_max - x_inter_min)
        inter_h = max(0, y_inter_max - y_inter_min)
        inter = inter_w * inter_h

        union = w1 * h1 + w1b * h1b - inter
        return inter / union if union > 0 else 0.0

    def match_detections(self, gt_boxes, pred_boxes):
        """Match predictions to ground truth via IoU"""
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return [], []

        matched_pairs = []
        used_gt = set()
        used_pred = set()

        # Greedy matching by highest IoU
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_j = -1
            for j, pred_box in enumerate(pred_boxes):
                if j in used_pred:
                    continue
                iou = self.iou_xywh(gt_box, pred_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                matched_pairs.append((i, best_j))
                used_gt.add(i)
                used_pred.add(best_j)

        tp = len(matched_pairs)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        return tp, fp, fn

    def compute_metrics(self, tp, fp, fn):
        """Compute Precision, Recall, MOTA"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mota = 1 - (fp + fn) / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall, mota

    def fuse_detections(self, yolo_dets, mog2_dets, flow_dets, weights):
        """
        Fuse detections from three sources using weighted confidence.

        Each detection: [x, y, w, h, confidence]
        weights: [w_yolo, w_mog2, w_flow]
        """
        w_yolo, w_mog2, w_flow = weights

        # Normalize weights
        total_w = w_yolo + w_mog2 + w_flow
        w_yolo, w_mog2, w_flow = w_yolo / total_w, w_mog2 / total_w, w_flow / total_w

        # Create detection pool with source-weighted confidence
        fused = {}

        # Add YOLO detections
        for x, y, w, h, conf in yolo_dets:
            key = (round(x, 1), round(y, 1), round(w, 1), round(h, 1))
            fused[key] = w_yolo * conf

        # Add MOG2 detections (MOG2 gives binary detections, treat as conf=0.8)
        for x, y, w, h in mog2_dets:
            key = (round(x, 1), round(y, 1), round(w, 1), round(h, 1))
            if key in fused:
                fused[key] += w_mog2 * 0.8
            else:
                fused[key] = w_mog2 * 0.8

        # Add Optical Flow detections (treat as conf=0.7)
        for x, y, w, h in flow_dets:
            key = (round(x, 1), round(y, 1), round(w, 1), round(h, 1))
            if key in fused:
                fused[key] += w_flow * 0.7
            else:
                fused[key] = w_flow * 0.7

        # Convert back to box format, apply confidence threshold
        result = []
        conf_threshold = 0.3  # Fused confidence threshold
        for (x, y, w, h), fused_conf in fused.items():
            if fused_conf >= conf_threshold:
                result.append([x, y, w, h, fused_conf])

        return np.array(result) if result else np.array([]).reshape(0, 5)

    def evaluate_on_sequence(self, gt_data, yolo_dets, mog2_dets, flow_dets, weights):
        """
        Evaluate ensemble with given weights on a sequence.

        Returns: tp, fp, fn, precision, recall, mota
        """
        total_tp, total_fp, total_fn = 0, 0, 0

        frames = sorted(set(gt_data.keys()) | set(yolo_dets.keys()))

        for frame in frames:
            gt_boxes = gt_data.get(frame, [])
            yolo_frame = yolo_dets.get(frame, [])
            mog2_frame = mog2_dets.get(frame, [])
            flow_frame = flow_dets.get(frame, [])

            # Fuse detections
            fused = self.fuse_detections(yolo_frame, mog2_frame, flow_frame, weights)

            if len(fused) > 0:
                pred_boxes = fused[:, :4]
            else:
                pred_boxes = []

            # Match and compute metrics
            tp, fp, fn = self.match_detections(gt_boxes, pred_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision, recall, mota = self.compute_metrics(total_tp, total_fp, total_fn)
        return total_tp, total_fp, total_fn, precision, recall, mota

    def optimize_weights(self, training_sequences):
        """
        Optimize ensemble weights using training sequences.

        training_sequences: list of {
            'gt': ground truth boxes by frame,
            'yolo': YOLO detections by frame,
            'mog2': MOG2 detections by frame,
            'flow': Optical flow detections by frame
        }
        """
        # Build training data for linear regression
        X_train = []
        y_train = []

        # Generate weight candidates
        weight_candidates = [
            [0.5, 0.3, 0.2],  # Original manual tuning
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.4, 0.4, 0.2],
            [0.5, 0.25, 0.25],
            [0.8, 0.1, 0.1],
            [0.6, 0.25, 0.15],
        ]

        print("Evaluating weight candidates on training data...")
        results_log = []

        for weights in weight_candidates:
            seq_motas = []

            for seq_idx, seq in enumerate(training_sequences):
                tp, fp, fn, prec, recall, mota = self.evaluate_on_sequence(
                    seq['gt'], seq['yolo'], seq['mog2'], seq['flow'], weights
                )
                seq_motas.append(mota)

            avg_mota = np.mean(seq_motas)
            X_train.append(weights)
            y_train.append(avg_mota)

            results_log.append({
                'weights': weights,
                'avg_mota': float(avg_mota),
                'seq_motas': [float(m) for m in seq_motas]
            })

            print(f"Weights {weights}: MOTA = {avg_mota:.4f}")

        # Fit linear regression (simple model for weight optimization)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Find best weights from candidates
        best_idx = np.argmax(y_train)
        self.optimal_weights = X_train[best_idx]
        best_mota = y_train[best_idx]

        print(f"\nOptimal weights found: {self.optimal_weights}")
        print(f"Best training MOTA: {best_mota:.4f}")

        return self.optimal_weights, results_log


def load_training_sequences(data_dir, num_sequences=20):
    """Load training sequence data from disk"""
    # This is a placeholder - in real scenario, load from your dataset
    sequences = []

    for seq_idx in range(num_sequences):
        seq = {
            'gt': {},  # Load from ground truth files
            'yolo': {},  # Load from YOLO predictions
            'mog2': {},  # Load from MOG2 detections
            'flow': {}  # Load from optical flow detections
        }
        # Load actual data from files...
        sequences.append(seq)

    return sequences


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Load training data (first 20 sequences)
    print("Loading training sequences...")
    train_sequences = load_training_sequences("data/", num_sequences=20)

    # Initialize optimizer
    optimizer = EnsembleWeightOptimizer(iou_threshold=0.1)

    # Optimize weights
    optimal_weights, results = optimizer.optimize_weights(train_sequences)

    # Save results
    with open("ensemble_weight_optimization_results.json", "w") as f:
        json.dump({
            'optimal_weights': optimal_weights.tolist(),
            'optimization_history': results
        }, f, indent=2)

    print("\nOptimization complete. Results saved to ensemble_weight_optimization_results.json")
