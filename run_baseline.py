"""
Generic baseline runner - works with any tracker that inherits from BaseTracker
Reads configuration from YAML files
"""
import argparse
import yaml
from pathlib import Path

from utils.data_loader import SMOT4SBDataset
from utils.evaluation import run_tracker_on_dataset


def merge_configs(base_config, experiment_config):
    """Merge base and experiment configs (experiment overrides base)"""
    merged = {**base_config}

    for key, value in experiment_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


def load_tracker(config):
    """
    Dynamically load tracker based on config

    Args:
        config: dict with tracker configuration

    Returns:
        tracker instance
    """
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
    # Add more detectors here as you implement them
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def run_experiment(config_file, overrides=None):
    """
    Run tracking experiment from config file

    Args:
        config_file: path to experiment config YAML
        overrides: dict of config values to override (e.g., from command line)
    """
    # Load configs
    config_path = Path(config_file)
    base_config_path = config_path.parent / 'base_config.yaml'

    print(f"Loading config: {config_file}")

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(config_file, 'r') as f:
        experiment_config = yaml.safe_load(f)

    # Merge configs
    config = merge_configs(base_config, experiment_config)

    # Apply command-line overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value

    print(f"\n{'=' * 60}")
    print(f"Experiment: {config['name']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = SMOT4SBDataset(
        config['data']['root'],
        config['data']['annotation_file']
    )

    # Initialize tracker
    print("Initializing tracker...")
    tracker = load_tracker(config)

    # Run evaluation
    all_metrics, avg_metrics = run_tracker_on_dataset(
        dataset=dataset,
        tracker=tracker,
        output_dir=config['output']['dir'],
        model_name=config['name'],
        max_videos=config['data'].get('max_videos'),
        visualize=config['output'].get('visualize', False)
    )

    return all_metrics, avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run tracking baseline from config file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run YOLO11n baseline
  python run_baseline.py --config configs/yolo11n_sort.yaml

  # Run with limited videos
  python run_baseline.py --config configs/yolo11n_sort.yaml --max_videos 5

  # Override confidence threshold
  python run_baseline.py --config configs/yolo11n_sort.yaml --set detector.conf_threshold=0.05

  # Enable visualization
  python run_baseline.py --config configs/yolo11n_sort.yaml --visualize
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config YAML file')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max videos to process (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization videos (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cpu or cuda (overrides config)')
    parser.add_argument('--set', type=str, action='append', dest='overrides',
                        help='Override config values (format: key.subkey=value)')

    args = parser.parse_args()

    # Build overrides dict
    overrides = {}

    if args.max_videos is not None:
        overrides['data.max_videos'] = args.max_videos

    if args.visualize:
        overrides['output.visualize'] = True

    if args.device is not None:
        overrides['device'] = args.device

    # Parse --set overrides
    if args.overrides:
        for override in args.overrides:
            key, value = override.split('=')
            # Try to parse value as int/float/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.lower() == 'null':
                        value = None

            overrides[key] = value

    # Run experiment
    all_metrics, avg_metrics = run_experiment(args.config, overrides)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
