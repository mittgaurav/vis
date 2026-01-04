#!/bin/bash
# Run all baselines per video (not per baseline)
# This allows easy comparison of results

# Configuration
MAX_VIDEOS=""  # to process all

# List of configs to run (only working baselines)
CONFIGS=(
    # exploratory
    "configs/motion_yolo_sort.yaml"
    "configs/yolo_tile_sort.yaml"
#    "configs/motion_yolo_dino_sort.yaml"
#    "configs/motion_multiscale_tracker.yaml"
#    "configs/raft_yolo_bytetrack.yaml"  # slow and poor results
#    "configs/ensemble_tracker.yaml"  # slow

    # baselines
    "configs/motion_sort.yaml"
    "configs/yolo_sort.yaml"
    "configs/yolo_ocsort.yaml"
    "configs/yolo_bytetrack.yaml"
    "configs/rtdetr_sort.yaml"
#    "configs/dino_sort.yaml"

#    "configs/clip_sort.yaml"  # too slow and requires GPU to run properly
#    "configs/centertrack.yaml"  # Extra setup and requires GPU to run properly
#    "configs/fairmot.yaml"  # Extra setup and requires GPU to run properly
)

echo "========================================"
echo "Running all baselines per video"
echo "Configs: ${#CONFIGS[@]}"
echo "Max videos: $MAX_VIDEOS"
echo "========================================"
echo ""

if [ -z "$MAX_VIDEOS" ]; then
    python run_baselines_per_video.py --configs "${CONFIGS[@]}"
else
    python run_baselines_per_video.py --configs "${CONFIGS[@]}" --max_videos $MAX_VIDEOS
fi

echo ""
echo "========================================"
echo "Complete!"
echo "Results in: results/per_video_baseline/"
echo "  - per_video_comparison/: Results per video"
echo "  - aggregate_results/: Summary across all videos"
echo "========================================"
