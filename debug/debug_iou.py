#!/usr/bin/env python3
import numpy as np
import cv2
from ultralytics import YOLO
from utils.data_loader import SMOT4SBDataset


def compute_iou(bbox1, bbox2):
    """bbox format: [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi = max(x1, x2)
    yi = max(y1, y2)
    xi_max = min(x1 + w1, x2 + w2)
    yi_max = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi_max - xi)
    inter_h = max(0, yi_max - yi)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


# Load YOLO and dataset
yolo = YOLO("../yolo12s.pt")
dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

# Test on first video, first frame
for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(dataset.get_video_ids()[0]):
    print(f"\n=== Frame {frame_id} ===")
    print(f"Image shape: {image.shape}")
    print(f"Ground truth: {len(gt_boxes)} birds")

    # Show GT boxes
    for i, (gt_box, gt_id) in enumerate(zip(gt_boxes, gt_ids)):
        x, y, w, h = gt_box
        print(f"  GT {i}: ID={gt_id}, bbox=[{x:.1f}, {y:.1f}, {w:.1f}x{h:.1f}], area={w * h:.0f}px")

    # Get YOLO detections
    results = yolo(image, conf=0.001, device='cpu', verbose=False)

    yolo_dets = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            yolo_dets.append([x, y, w, h, conf, cls])

    print(f"\nYOLO detections: {len(yolo_dets)}")
    for i, det in enumerate(yolo_dets[:5]):  # Show first 5
        x, y, w, h, conf, cls = det
        print(f"  YOLO {i}: class={cls}, conf={conf:.4f}, bbox=[{x:.1f}, {y:.1f}, {w:.1f}x{h:.1f}], area={w * h:.0f}px")

    # Compute IoU between GT and YOLO
    print(f"\nIoU Matrix (GT vs YOLO):")
    if len(yolo_dets) > 0 and len(gt_boxes) > 0:
        print("     ", end="")
        for j in range(min(5, len(yolo_dets))):
            print(f"YOLO{j:2d}  ", end="")
        print()

        for i, gt_box in enumerate(gt_boxes):
            print(f"GT{i}: ", end="")
            for j, yolo_det in enumerate(yolo_dets[:5]):
                iou = compute_iou(gt_box, yolo_det[:4])
                print(f"{iou:.3f}  ", end="")
            print()

        # Find best matches
        print(f"\nBest IoU for each GT:")
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_j = -1
            for j, yolo_det in enumerate(yolo_dets):
                iou = compute_iou(gt_box, yolo_det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            print(f"  GT {i}: best match YOLO {best_j} with IoU={best_iou:.4f}")

    break
