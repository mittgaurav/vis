import json
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

from utils.data_loader import SMOT4SBDataset
from utils.evaluation import evaluate_and_save_video


def merge_configs(base_config, experiment_config):
    """Merge base and experiment configs"""
    merged = {**base_config}
    for key, value in experiment_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_all_configs(config_files, base_config_path):
    """Load and merge all experiment configs"""
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    configs = []
    for config_file in config_files:
        with open(config_file, "r") as f:
            exp_config = yaml.safe_load(f)
        merged = merge_configs(base_config, exp_config)
        configs.append((exp_config["name"], merged))

    return configs


def load_tracker(config):
    """Load tracker from config"""
    detector_type = config["detector"]["type"].lower()
    # Dynamically import the correct tracker based on the detector type
    tracker_map = {
        "yolo_sort": "baselines.yolo_sort.YOLOSORT",
        "yolo_ocsort": "baselines.yolo_ocsort.YOLOOCSORT",
        "yolo_bytetrack": "baselines.yolo_bytetrack.YOLOByteTrack",
        "rfdetr_sort": "baselines.rfdetr_sort.RFDETRSORT",
        "clip_sort": "baselines.clip_sort.CLIPSORT",
        "dino_sort": "baselines.dino_sort.DINOSORT",
        "motion_sort": "baselines.motion_sort.MotionSORT",
        "centertrack": "baselines.centertrack.CenterTracker",
        "fairmot": "baselines.fairmot.FairMOTTracker",
        "motion_yolo_tracker": "exploratory.motion_yolo_tracker.MotionYOLOTracker",
        "motion_yolo_dino_sort": "exploratory.motion_yolo_dino_sort.MotionYOLODINOTracker",
        "motion_multiscale": "exploratory.motion_multiscale_tracker.MotionMultiScaleTracker",
        "raft_dino": "exploratory.raft_dino_tracker.RAFTDINOTracker",
        "ensemble": "exploratory.ensemble_tracker.EnsembleTracker",
    }

    if detector_type not in tracker_map:
        raise ValueError(f"Unknown detector type: {detector_type}")

    module_path, class_name = tracker_map[detector_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    tracker_class = getattr(module, class_name)

    return tracker_class(config)


def run_per_video(config_files, output_dir, max_videos=None):
    """
    Run all baselines on each video sequentially
    """
    print(f"\n{'=' * 80}")
    print("RUNNING BASELINES PER VIDEO")
    print(f"Configs: {len(config_files)}")
    print(f"{'=' * 80}\n")

    base_config_path = Path(config_files[0]).parent / "base_config.yaml"

    configs = load_all_configs(config_files, base_config_path)

    print(f"Loaded {len(configs)} baseline configurations:")
    for name, _ in configs:
        print(f"  - {name}")
    print()

    first_config = configs[0][1]
    dataset = SMOT4SBDataset(first_config["data"]["root"], first_config["data"]["annotation_file"])

    video_ids = dataset.get_video_ids()
    if max_videos:
        video_ids = video_ids[:max_videos]

    print(f"Processing {len(video_ids)} videos\n")
    per_video_results = {}

    for video_idx, video_id in enumerate(tqdm(video_ids, desc="Videos", unit="video"), 1):
        video_name = dataset.videos[video_id]["name"]
        print(f"\n{'=' * 80}")
        print(f"VIDEO {video_idx}/{len(video_ids)}: {video_name} (ID: {video_id})")
        print(f"{'=' * 80}")

        per_video_results[video_id] = {"video_name": video_name, "baselines": {}}

        for baseline_name, config in configs:
            print(f"\n  Running: {baseline_name}")

            tracker = load_tracker(config)
            predictions, stats = tracker.track_video(dataset, video_id)
            metrics = evaluate_and_save_video(
                dataset,
                video_id,
                video_name,
                predictions,
                stats,
                output_dir,
                baseline_name,
                video_idx,
                len(video_ids),
                visualize=config["output"].get("visualize", False),
            )

            per_video_results[video_id]["baselines"][baseline_name] = {
                "metrics": metrics,
                "stats": stats,
            }

            mota = metrics.get("mota", 0)
            precision = metrics.get("precision", 0)
            hota = metrics.get("hota", 0)
            print(f"    ✓ {baseline_name}: " f"MOTA={mota:.4f}, " f"Precision={precision:.4f}, " f"HOTA={hota:.4f}, " f"FPS={stats['avg_fps']:.2f}")

        save_per_video_comparison(per_video_results[video_id], output_dir, video_id)
        save_aggregate_results(per_video_results, output_dir, configs)

    print(f"\n{'=' * 80}")
    print("ALL BASELINES COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


def save_per_video_comparison(video_results, output_dir, video_id):
    """Save comparison of all baselines for a single video"""
    output_dir = Path(output_dir) / "per_video_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"video_{video_id}_comparison.json"
    with open(json_file, "w") as f:
        json.dump(video_results, f, indent=2, default=str)

    rows = []
    for baseline_name, result in video_results["baselines"].items():
        if "error" in result:
            continue
        metrics = result["metrics"]
        stats = result["stats"]
        # MINIMAL CHANGE: add hota to per-video CSV
        row = {
            "baseline": baseline_name,
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "mota": metrics.get("mota", 0),
            "motp": metrics.get("motp", 0),
            "hota": metrics.get("hota", 0),
            "dotd": metrics.get("dotd", float("inf")),
            "num_switches": metrics.get("num_switches", 0),
            "fps": stats.get("avg_fps", 0),
            "avg_detections": stats.get("avg_detections_per_frame", 0),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_file = output_dir / f"video_{video_id}_comparison.csv"
        df.to_csv(csv_file, index=False)


def save_aggregate_results(per_video_results, output_dir, configs):
    """Save aggregate results across all videos for each model"""
    output_dir = Path(output_dir) / "aggregate_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_aggregates = {}
    for video_id, video_result in per_video_results.items():
        for baseline_name, result in video_result["baselines"].items():
            if "error" in result:
                continue
            if baseline_name not in baseline_aggregates:
                baseline_aggregates[baseline_name] = {"metrics": [], "stats": []}
            baseline_aggregates[baseline_name]["metrics"].append(result["metrics"])
            baseline_aggregates[baseline_name]["stats"].append(result["stats"])

    summary = {}
    for baseline_name, data in baseline_aggregates.items():
        metrics_list = data["metrics"]
        if not metrics_list:
            continue

        avg_metrics = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0

        stats_list = data["stats"]
        avg_stats = {}
        for key in stats_list[0].keys():
            if isinstance(stats_list[0][key], (int, float)):
                values = [s[key] for s in stats_list if key in s]
                avg_stats[key] = sum(values) / len(values) if values else 0

        summary[baseline_name] = {
            "num_videos": len(metrics_list),
            "avg_metrics": avg_metrics,
            "avg_stats": avg_stats,
        }

    json_file = output_dir / "aggregate_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    rows = []
    for baseline_name, data in summary.items():
        avg_m = data["avg_metrics"]
        avg_s = data["avg_stats"]
        # MINIMAL CHANGE: include hota in aggregate CSV
        row = {
            "baseline": baseline_name,
            "num_videos": data["num_videos"],
            "precision": avg_m.get("precision", 0),
            "recall": avg_m.get("recall", 0),
            "mota": avg_m.get("mota", 0),
            "motp": avg_m.get("motp", 0),
            "hota": avg_m.get("hota", 0),  # new
            "dotd": avg_m.get("dotd", float("inf")),
            "num_switches": avg_m.get("num_switches", 0),
            "fps": avg_s.get("avg_fps", 0),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values("mota", ascending=False)

        csv_file = output_dir / "aggregate_comparison.csv"
        df.to_csv(csv_file, index=False)

        print(f"\n{'=' * 80}")
        print("AGGREGATE RESULTS ACROSS ALL VIDEOS")
        print(f"{'=' * 80}")
        print(df.to_string(index=False))
        print(f"\nSaved to: {csv_file}")
