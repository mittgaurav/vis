#!/usr/bin/env python3
import numpy as np
from ultralytics import YOLO
from utils.data_loader import SMOT4SBDataset

# Load YOLO
print("Loading YOLO12s...")
yolo = YOLO("../yolo12s.pt")

# Load dataset
dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

# Get first video
video_ids = dataset.get_video_ids()
print(f"Testing on video: {video_ids[0]}")

frame_count = 0
detection_counts = []

for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_ids[0]):
    frame_count += 1

    # Try different confidence thresholds
    for conf_thresh in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        results = yolo(image, conf=conf_thresh, device='cpu', verbose=False)

        bird_count = 0
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 14:  # Bird class
                    bird_count += 1

        if frame_count == 1:
            print(f"Frame 1 @ conf={conf_thresh}: {bird_count} birds, {len(gt_boxes)} GT")
            if bird_count > 0:
                for result in results:
                    for i, box in enumerate(result.boxes):
                        if int(box.cls[0]) == 14:
                            print(
                                f"  Bird {i}: conf={float(box.conf[0]):.4f}, size={(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]):.0f}px")

    # Use conf=0.001 for counting
    results = yolo(image, conf=0.001, device='cpu', verbose=False)
    bird_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 14:
                bird_count += 1
    detection_counts.append(bird_count)

    if frame_count == 5:
        break

print(f"\nFirst 5 frames detection counts: {detection_counts}")
print(f"Average: {np.mean(detection_counts):.2f} birds/frame")
print(f"Ground truth: {len(gt_boxes)} birds in this frame")

# If we're getting detections, check their confidence distribution
if np.sum(detection_counts) > 0:
    print("\n✅ YOLO IS DETECTING BIRDS")
    print("The issue is with confidence thresholds in config")
else:
    print("\n❌ YOLO IS NOT DETECTING ANY BIRDS")
    print("Possible causes:")
    print("  1. YOLO model weights not loaded correctly")
    print("  2. Bird class ID is not 14 (check your dataset)")
    print("  3. Image format issue (BGR vs RGB)")
