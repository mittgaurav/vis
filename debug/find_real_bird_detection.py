#!/usr/bin/env python3
import numpy as np
import cv2
from ultralytics import YOLO
from utils.data_loader import SMOT4SBDataset


def compute_iou(bbox1, bbox2):
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


yolo = YOLO("../yolo12s.pt")
dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(dataset.get_video_ids()[0]):
    gt_box = gt_boxes[0]  # First GT bird
    print(f"GT bird: {gt_box}")

    results = yolo(image, conf=0.001, device='cpu', verbose=False)

    best_iou = -1
    best_det = None
    best_cls = None

    # Check ALL detections to see which is closest to GT
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            iou = compute_iou(gt_box, [x, y, w, h])

            if iou > best_iou:
                best_iou = iou
                best_det = [x, y, w, h, conf]
                best_cls = cls

    print(f"\nBest matching YOLO detection:")
    print(f"  Class: {best_cls}")
    print(f"  Box: {best_det[:4]}")
    print(f"  Conf: {best_det[4]:.4f}")
    print(f"  IoU: {best_iou:.4f}")

    if best_iou > 0:
        print(f"\n✓ Found it! Use filter_class: {best_cls}")
    else:
        print(f"\n✗ YOLO didn't detect this bird at all")
        print("\nAll YOLO detections:")
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                iou = compute_iou(gt_box, [x, y, w, h])
                print(f"  Class {cls}: conf={conf:.4f}, IoU={iou:.6f}")

    break
