"""
Run all baselines per video (not per baseline)
This allows easier comparison of results across methods for each video
"""
import argparse
import yaml
import json
from pathlib import Path
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
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    configs = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            exp_config = yaml.safe_load(f)
        merged = merge_configs(base_config, exp_config)
        configs.append((exp_config['name'], merged))

    return configs


def load_tracker(config):
    """Load tracker from config"""
    detector_type = config['detector']['type'].lower()

    if detector_type == 'yolo':
        from baselines.yolo_sort import YOLOSORTTracker
        return YOLOSORTTracker(config)
    elif detector_type == 'rfdetr':
        from baselines.rfdetr_sort import RFDETRSORTTracker
        return RFDETRSORTTracker(config)
    elif detector_type == 'clip':
        from baselines.clip_sort import CLIPSORTTracker
        return CLIPSORTTracker(config)
    elif detector_type == 'dino':
        from baselines.dino_sort import DINOSORTTracker
        return DINOSORTTracker(config)
    elif detector_type == 'centertrack':
        from baselines.centertrack import CenterTracker
        return CenterTracker(config)
    elif detector_type == 'fairmot':
        from baselines.fairmot import FairMOTTracker
        return FairMOTTracker(config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def run_per_video(config_files, output_dir, max_videos=None):
    """
    Run all baselines on each video sequentially

    Args:
        config_files: list of config file paths
        output_dir: base output directory
        max_videos: max videos to process (None = all)
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING BASELINES PER VIDEO")
    print(f"Configs: {len(config_files)}")
    print(f"{'=' * 80}\n")

    # Load base config
    base_config_path = Path(config_files[0]).parent / 'base_config.yaml'

    # Load all configs
    configs = load_all_configs(config_files, base_config_path)

    print(f"Loaded {len(configs)} baseline configurations:")
    for name, _ in configs:
        print(f"  - {name}")
    print()

    # Load dataset (only once)
    first_config = configs[0][1]
    dataset = SMOT4SBDataset(
        first_config['data']['root'],
        first_config['data']['annotation_file']
    )

    # Get video IDs
    video_ids = dataset.get_video_ids()
    if max_videos:
        video_ids = video_ids[:max_videos]

    print(f"Processing {len(video_ids)} videos\n")

    # Store results per video
    per_video_results = {}

    # Process each video
    for video_idx, video_id in enumerate(tqdm(video_ids, desc="Videos", unit="video"), 1):
        video_name = dataset.videos[video_id]['name']

        print(f"\n{'=' * 80}")
        print(f"VIDEO {video_idx}/{len(video_ids)}: {video_name} (ID: {video_id})")
        print(f"{'=' * 80}")

        per_video_results[video_id] = {
            'video_name': video_name,
            'baselines': {}
        }

        # Run each baseline on this video
        for baseline_name, config in configs:
            print(f"\n  Running: {baseline_name}")

            try:
                # Initialize tracker
                tracker = load_tracker(config)

                # Track video
                predictions, stats = tracker.track_video(dataset, video_id)

                # Evaluate and save
                metrics = evaluate_and_save_video(
                    dataset, video_id, video_name, predictions, stats,
                    output_dir, baseline_name, video_idx, len(video_ids),
                    visualize=config['output'].get('visualize', False)
                )

                # Store results
                per_video_results[video_id]['baselines'][baseline_name] = {
                    'metrics': metrics,
                    'stats': stats
                }

                print(f"    ✓ {baseline_name}: MOTA={metrics.get('mota', 0):.4f}, "
                      f"Precision={metrics.get('precision', 0):.4f}, "
                      f"FPS={stats['avg_fps']:.2f}")

            except Exception as e:
                print(f"    ✗ {baseline_name} FAILED: {e}")
                per_video_results[video_id]['baselines'][baseline_name] = {
                    'error': str(e)
                }

        # Save per-video comparison
        save_per_video_comparison(per_video_results[video_id], output_dir, video_id)

    # Save aggregate results
    save_aggregate_results(per_video_results, output_dir, configs)

    print(f"\n{'=' * 80}")
    print("ALL BASELINES COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


def save_per_video_comparison(video_results, output_dir, video_id):
    """Save comparison of all baselines for a single video"""
    output_dir = Path(output_dir) / "per_video_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_dir / f"video_{video_id}_comparison.json"
    with open(json_file, 'w') as f:
        json.dump(video_results, f, indent=2, default=str)

    # Create CSV for easy viewing
    rows = []
    for baseline_name, result in video_results['baselines'].items():
        if 'error' in result:
            continue

        metrics = result['metrics']
        stats = result['stats']

        row = {
            'baseline': baseline_name,
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'mota': metrics.get('mota', 0),
            'motp': metrics.get('motp', 0),
            'dotd': metrics.get('dotd', float('inf')),
            'num_switches': metrics.get('num_switches', 0),
            'fps': stats.get('avg_fps', 0),
            'avg_detections': stats.get('avg_detections_per_frame', 0)
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_file = output_dir / f"video_{video_id}_comparison.csv"
        df.to_csv(csv_file, index=False)


def save_aggregate_results(per_video_results, output_dir, configs):
    """Save aggregate results across all videos"""
    output_dir = Path(output_dir) / "aggregate_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metrics per baseline
    baseline_aggregates = {}

    for video_id, video_result in per_video_results.items():
        for baseline_name, result in video_result['baselines'].items():
            if 'error' in result:
                continue

            if baseline_name not in baseline_aggregates:
                baseline_aggregates[baseline_name] = {
                    'metrics': [],
                    'stats': []
                }

            baseline_aggregates[baseline_name]['metrics'].append(result['metrics'])
            baseline_aggregates[baseline_name]['stats'].append(result['stats'])

    # Compute averages
    summary = {}

    for baseline_name, data in baseline_aggregates.items():
        metrics_list = data['metrics']

        if not metrics_list:
            continue

        # Average metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0

        # Average stats
        stats_list = data['stats']
        avg_stats = {}
        for key in stats_list[0].keys():
            if isinstance(stats_list[0][key], (int, float)):
                values = [s[key] for s in stats_list if key in s]
                avg_stats[key] = sum(values) / len(values) if values else 0

        summary[baseline_name] = {
            'num_videos': len(metrics_list),
            'avg_metrics': avg_metrics,
            'avg_stats': avg_stats
        }

    # Save JSON
    json_file = output_dir / "aggregate_summary.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Create comparison CSV
    rows = []
    for baseline_name, data in summary.items():
        row = {
            'baseline': baseline_name,
            'num_videos': data['num_videos'],
            'precision': data['avg_metrics'].get('precision', 0),
            'recall': data['avg_metrics'].get('recall', 0),
            'mota': data['avg_metrics'].get('mota', 0),
            'motp': data['avg_metrics'].get('motp', 0),
            'dotd': data['avg_metrics'].get('dotd', float('inf')),
            'num_switches': data['avg_metrics'].get('num_switches', 0),
            'fps': data['avg_stats'].get('avg_fps', 0)
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values('mota', ascending=False)

        csv_file = output_dir / "aggregate_comparison.csv"
        df.to_csv(csv_file, index=False)

        print(f"\n{'=' * 80}")
        print("AGGREGATE RESULTS ACROSS ALL VIDEOS")
        print(f"{'=' * 80}")
        print(df.to_string(index=False))
        print(f"\nSaved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run all baselines per video (not per baseline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines on first 5 videos
  python run_baselines_per_video.py \\
      --configs configs/yolo12n_sort.yaml configs/rfdetr_sort.yaml configs/dino_sort.yaml \\
      --max_videos 5

  # Run on all videos
  python run_baselines_per_video.py \\
      --configs configs/*.yaml
        """
    )

    parser.add_argument('--configs', nargs='+', required=True,
                        help='List of config files to run')
    parser.add_argument('--output_dir', type=str, default='results/per_video_baseline',
                        help='Output directory')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max videos to process')

    args = parser.parse_args()

    run_per_video(args.configs, args.output_dir, args.max_videos)


if __name__ == "__main__":
    main()
