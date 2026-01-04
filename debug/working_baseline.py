#!/usr/bin/env python3

import yaml
from baselines.yolo_sort import YOLOSORT
from utils.data_loader import SMOT4SBDataset
from utils.evaluation import run_tracker_on_dataset

# Load configs
with open("working_baseline.yaml") as f:
    config = yaml.safe_load(f)
with open("../configs/base_config.yaml") as f:
    base_config = yaml.safe_load(f)

# Merge
config = {**base_config, **config, "detector": config["detector"], "tracker": config["tracker"]}

# Load dataset
dataset = SMOT4SBDataset(config["data"]["root"], config["data"]["annotation_file"])

# Test on first 5 videos only
print("Testing YOLO+SORT on 5 videos...")
tracker = YOLOSORT(config)

all_metrics, avg_metrics = run_tracker_on_dataset(
    dataset,
    tracker,
    output_dir="results/working_baseline_test",
    model_name="working_baseline",
    max_videos=5,
    visualize=False
)

print("\n✅ Test complete. Check results/working_baseline_test/")
print(f"   MOTA: {avg_metrics.get('mota', 0):.4f}")
print(f"   Precision: {avg_metrics.get('precision', 0):.4f}")
print(f"   Recall: {avg_metrics.get('recall', 0):.4f}")
