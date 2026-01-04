#!/usr/bin/env python3
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

frame_count = 0
found_count = 0
no_gt_count = 0

print("Testing YOLO on first 50 frames...\n")

for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(dataset.get_video_ids()[4]):
    frame_count += 1

    if frame_count > 600:
        break

    # Skip frames with no GT
    if len(gt_boxes) == 0:
        no_gt_count += 1
        continue

    # Get YOLO detections
    results = yolo(image, conf=0.001, device='cpu', verbose=False)

    # Find best match for first GT bird
    gt_box = gt_boxes[0]
    best_iou = 0
    best_det = None
    best_cls = None

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

    # Report
    if best_iou > 0.001:
        found_count += 1
        print(f"Frame {frame_id}: ✓ IoU={best_iou:.3f}, Class={best_cls}, Conf={best_det[4]:.4f}")
    else:
        # Print first 10 frames regardless
        print(f"Frame {frame_id}: ✗ IoU={best_iou:.6f} (GT: {gt_box[:2].astype(int)})")

print(f"\n=== SUMMARY ===")
print(f"Frames tested: {frame_count}")
print(f"Frames with no GT: {no_gt_count}")
print(f"Frames where YOLO found bird (IoU > 0.001): {found_count}/{frame_count - no_gt_count}")
print(f"Success rate: {100 * found_count / (frame_count - no_gt_count):.1f}%")

if found_count == 0:
    print("\n❌ YOLO CANNOT DETECT BIRDS IN THIS DATASET")
    print("→ Use motion detection (background subtraction) instead")
else:
    print(f"\n✓ YOLO CAN detect some birds")
    print(f"→ Use filter_class: {best_cls} with low IoU threshold")
