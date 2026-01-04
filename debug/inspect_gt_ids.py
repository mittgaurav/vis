#!/usr/bin/env python3
import json
from utils.data_loader import SMOT4SBDataset

dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

# Inspect the raw COCO data structure
print("=== COCO Data Structure ===\n")

# Check what's in the annotations
print("Annotation keys:", dataset.coco_data['annotations'][0].keys())
print("\nSample annotation:")
print(json.dumps(dataset.coco_data['annotations'][0], indent=2))

print("\n\n=== GT IDs Analysis ===\n")

video_id = dataset.get_video_ids()[0]
print(f"Video {video_id}:")

frame_count = 0
for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_id):
    frame_count += 1

    if frame_count <= 5:
        print(f"\nFrame {frame_id}:")
        print(f"  GT boxes: {gt_boxes}")
        print(f"  GT IDs: {gt_ids}")

        # Show raw annotations for this frame
        frames = dataset.get_video_frames(video_id)
        image_info = [f for f in frames if f['frame_id'] == frame_id][0]
        anns = dataset.get_frame_annotations(image_info['id'])

        print(f"  Raw annotations: {len(anns)}")
        for ann in anns:
            print(
                f"    - bbox: {ann['bbox']}, track_id: {ann.get('track_id', 'MISSING')}, category_id: {ann.get('category_id', 'MISSING')}")

    if frame_count >= 5:
        break

print("\n\n=== What Are track_id Values? ===\n")

# Get all unique track IDs in the video
all_track_ids = set()
for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_id):
    for tid in gt_ids:
        all_track_ids.add(tid)

print(f"Unique track IDs in video {video_id}: {sorted(all_track_ids)}")
print(f"Total unique birds: {len(all_track_ids)}")

print("\n\n=== What About YOLO IDs? ===\n")

from ultralytics import YOLO

yolo = YOLO("../yolo12s.pt")

for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_id):
    print(f"\nFrame {frame_id}:")
    print(f"  GT track IDs: {list(gt_ids)}")

    results = yolo(image, conf=0.001, device='cpu', verbose=False)
    yolo_classes = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            yolo_classes.add(cls)

    print(f"  YOLO detections - classes found: {sorted(yolo_classes)}")
    print(f"  → These are COCO class IDs (0=person, 1=bicycle, 14=bird, etc)")
    print(f"  → They're NOT the same as GT track_ids!")

    break
