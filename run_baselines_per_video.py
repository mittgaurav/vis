import argparse
from utils.run import run_per_video

def main():
    parser = argparse.ArgumentParser(
        description='Run all baselines per video',
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
