#!/usr/bin/env python3
import numpy as np
from ultralytics import YOLO
from utils.data_loader import SMOT4SBDataset

yolo = YOLO("../yolo12s.pt")
dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

# Test on first frame
for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(dataset.get_video_ids()[0]):
    print(f"Frame {frame_id}: {len(gt_boxes)} ground truth birds at:")
    for i, box in enumerate(gt_boxes):
        x, y, w, h = box
        print(f"  Bird {i}: [{x:.0f}, {y:.0f}, {w:.0f}x{h:.0f}]")

    results = yolo(image, conf=0.001, device='cpu', verbose=False)

    # Group detections by class
    detections_by_class = {}
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            conf = float(box.conf[0])
            detections_by_class[cls].append({
                'bbox': [x, y, w, h],
                'conf': conf,
                'pos_str': f"[{x:.0f}, {y:.0f}]"
            })

    print(f"\nYOLO Detections by class:")
    for cls in sorted(detections_by_class.keys()):
        dets = detections_by_class[cls]
        print(f"  Class {cls}: {len(dets)} detections")
        for i, det in enumerate(dets[:3]):
            print(f"    {i}: {det['pos_str']}, conf={det['conf']:.4f}")

        # Check if any are near GT
        for det in dets[:5]:
            for gt_box in gt_boxes:
                gt_x, gt_y = gt_box[0], gt_box[1]
                det_x, det_y = det['bbox'][0], det['bbox'][1]
                dist = np.sqrt((gt_x - det_x) ** 2 + (gt_y - det_y) ** 2)
                if dist < 100:  # Within 100 pixels
                    print(f"    ✓ Close to GT! Distance={dist:.0f}px")

    break
