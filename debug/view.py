#!/usr/bin/env python3
import cv2
import sys
from utils.data_loader import SMOT4SBDataset

dataset = SMOT4SBDataset("../data/phase_1/train", "data/annotations/train.json")

# Parse command line args
if len(sys.argv) < 2:
    print("Usage: python visualize_gt_flexible.py <video_id> [frame_id]")
    print("Example: python visualize_gt_flexible.py 1")
    print("Example: python visualize_gt_flexible.py 1 5")
    sys.exit(1)

video_id = int(sys.argv[1])
target_frame_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

print(f"Loading video {video_id}...")

# Get all video IDs and check if valid
all_video_ids = dataset.get_video_ids()
if video_id not in all_video_ids:
    print(f"Invalid video ID. Available: {all_video_ids}")
    sys.exit(1)

frame_count = 0
found = False

for frame_id, image, gt_boxes, gt_ids in dataset.iterate_video(video_id):
    frame_count += 1

    # If target frame specified, only process that one
    if target_frame_id is not None and frame_id != target_frame_id:
        continue

    found = True

    print(f"\nFrame ID: {frame_id}")
    print(f"Image shape: {image.shape}")
    print(f"Ground truth birds: {len(gt_boxes)}")

    # Draw GT boxes
    img_copy = image.copy()
    for i, (bbox, track_id) in enumerate(zip(gt_boxes, gt_ids)):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw rectangle (green for GT)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label with more info
        label = f"ID:{int(track_id)} #{i}"
        cv2.putText(img_copy, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        print(f"  Bird {i}: ID={int(track_id)}, bbox=[{x}, {y}, {w}x{h}], area={w * h}px")

    # Save
    output_path = f"gt_video{video_id}_frame{frame_id}.jpg"
    cv2.imwrite(output_path, img_copy)
    print(f"\nSaved: {output_path}")

    # Display
    display = cv2.resize(img_copy, (1280, 720))
    cv2.imshow(f"Video {video_id}, Frame {frame_id} - Press any key", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If target frame specified, stop after showing it
    if target_frame_id is not None:
        break

if not found:
    if target_frame_id is not None:
        print(f"Frame {target_frame_id} not found in video {video_id}")
    else:
        print(f"Video {video_id} has no frames?")
else:
    print(f"\nTotal frames in video: {frame_count}")
