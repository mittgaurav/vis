# Small Flying Bird Tracking: Domain-Aware Detection and Tracking

## Project Overview

This repository contains a systematic approach to tracking small flying birds in video sequences under CPU-only computational constraints. The primary challenge is detecting and tracking tiny objects (30×30 pixels on average) that move rapidly (20-50 px/frame) against moving camera backgrounds.

### Problem Statement

Standard object detection and tracking methods fail catastrophically on small bird tracking due to:
- **Distribution mismatch**: COCO-trained detectors optimize for ~100×100 px objects; birds are 10× smaller
- **Confidence unreliability**: Standard confidence thresholds (0.5) discard true positives at extreme scales
- **Motion ambiguity**: Camera pan-tilt motion makes background subtraction infeasible
- **Computational constraints**: Real-time inference must operate on CPU (5-10 FPS target)

### Our Approach

Rather than proposing novel architectures, we take a principled engineering approach:

1. **Problem Analysis**: Identify root causes of standard method failure
2. **Targeted Adaptations**: Develop domain-aware parameter modifications:
   - **Confidence calibration**: Lower threshold to 0.005 (100× more permissive)
   - **Motion filtering**: Use MOG2 as secondary consistency check, not primary detector
   - **Spatial tiling**: Process 512×512 tiles to increase relative object size (0.4% → 8.8% of tile width)
   - **Adaptive tracking parameters**: Adjust for fast motion (max_age=10, iou_threshold=0.01)

3. **Systematic Ablation**: Evaluate multiple detector-tracker combinations to isolate bottlenecks

### Key Results
**Key Finding**: Spatial tiling improves MOTA through relative object size scaling, while maintaining real-time CPU operation.

---

## Dataset: SMOT4SB

The Small Multi-Object Tracking for Spotting Birds (SMOT4SB) dataset contains:
- **96 training sequences** of flying birds
- **Phase 1**: 3840×2160 resolution
- **Phase 2**: 1920×1080 resolution (used for evaluation)
- **Object sizes**: 10×10 to 30×30 pixels (average: 44.8×39.9 px)
- **Ground truth**: Bounding boxes with consistent track IDs

Download: https://drive.google.com/drive/folders/1Y1J13W6VlgDh-L28n_mVbs7HIfo_Hv5s into data folder

Or use the shared script
Setup (one-time):

1. Go to Google Cloud Console

2. Create project → Enable Drive API → Create credentials (OAuth 2.0)

3. Download credentials.json

```bash
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2 tqdm
```
Run
```bash
python download_vis_phase1.py
```

Put data in data/ as shown in the below structure.

---

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_baselines_per_video.py         # Main script: evaluate baselines on SMOT4SB
├── figures/                          # Result figures
│   └── *.png
│
├── baselines/                         # Baseline detector-tracker implementations
│   ├── yolo_sort.py                  # YOLO + SORT tracker
│   ├── yolo_ocsort.py                # YOLO + OC-SORT tracker
│   ├── yolo_bytetrack.py             # YOLO + ByteTrack tracker
│   ├── rtdetr_sort.py                # RT-DETR + SORT tracker
│   ├── motion_sort.py                # MOG2 background subtraction + SORT
│   └── base_tracker.py               # Base class for trackers
│
├── exploratory/                       # Explored but rejected approaches
│   ├── ensemble_tracker.py           # Multi-detector ensemble (rejected: too slow)
│   ├── yolo_tile_sort.py            # Spatial tiling (BEST - our approach)
│   ├── motion_yolo_sort.py          # Motion-filtered YOLO (rejected: poor precision)
│   └── reject/                       # Other rejected experiments
│       ├── raft_yolo_bytetrack.py   # Optical flow + YOLO (rejected: computational overhead)
│       └── optical_dino_tracker.py  # DINO features (rejected: minimal gain)
│
├── detectors/                         # Detection implementations
│   ├── yolo.py                       # YOLOv8/v11/v12 wrapper
│   ├── rtdetr.py                     # RT-DETR wrapper
│   └── __init__.py
│
├── trackers/                          # Tracking algorithms
│   ├── sort.py                       # SORT: Simple Online and Realtime Tracking
│   ├── ocsort.py                     # OC-SORT: Observation-centric SORT
│   ├── bytetrack.py                  # ByteTrack: Two-stage matching
│   └── __init__.py
│
├── utils/                             # Utility functions
│   ├── data_loader.py                # SMOT4SB dataset loader
│   ├── metrics.py                    # Evaluation metrics (Precision, Recall, MOTA, HOTA)
│   ├── evaluation.py                 # TrackEval integration
│   ├── hota_trackeval.py            # HOTA metric computation
│   ├── visualization.py              # Visualization utilities
│   ├── run.py                        # Core evaluation loop
│   └── __init__.py
│
├── data/                             # data
│   ├── annotations/
│   ├── phase_1/
│   │   └── train/
│   │       └── 0001/
│   │           └── 0001.jpg # and so on
│   └── phase_2/
│       └── train/
│           └── 0001/
│               └── 0001.jpg # and so on│
├── debug/                             # debug and visualization functions
│   ├── result_plots.py                # plot results
│   ├── *.py
│   └── __init__.py
│
└── results/                           # Output directory (created at runtime)
    └── per_video_baseline/            # Per-video results
        ├── yolo_tile_sort/
        ├── yolo_sort/
        └── ...

```

---

## Installation

### Requirements
- Python 3.8+ (preferrably 3.10)
- CPU or GPU (GPU optional, code runs on CPU)
- ~100 GB disk space for SMOT4SB dataset

### Step 1: Clone repository and install dependencies

```bash
git clone <repository_url>
cd vis
pip install -r requirements.txt
```

### Step 2: Download SMOT4SB dataset

```bash
# Download from Google Drive (link above) and extract
# this is a bit tortuous so please bear with yourself
unzip SMOT4SB.zip -d data/
```

Dataset structure:
```
data/
├── phase_1/
│   └── train/       # 3840×2160 resolution videos
│   └── pub_test/    # 3840×2160 resolution videos
├── phase_2/      # 1920×1080 resolution videos (used for evaluation)
│   └── train/
│   └── pub_test/
├── annotations/
│   └── train.json   # Ground truth for training split
```

We have added data_ with examples on how to structure the data. However, as the annotations are available only for both phase_1 and phase_2 train splits together, We found it easiest to
1. Copy entire phase_2/train content into phase_1/train
2. Rely only on train data for the evaluation (Except for the ensemble weight training where use the phase_1 train for training and phase_2 train for evaluation)

---

## Running the Code
### Shell script
Choose the baselines you want to run in **run_all_baselines_per_video.sh** and then run

```bash
sh ./run_all_baselines_per_video.sh
```

### Evaluate Single Baseline

```bash
# Evaluate YOLO Tiled + SORT (our best approach) on 5 videos
python run_baselines_per_video.py \
    --baseline yolo_tile_sort \
    --data_dir data/SMOT4SB/phase2 \
    --num_videos 5 \
    --output_dir results/
```

### Run All Baselines

```bash
python run_baselines_per_video.py \
    --data_dir data/SMOT4SB/phase2 \
    --num_videos 96 \
    --output_dir results/
```

### Run Specific Baseline

```bash
# Options: yolo_sort, yolo_ocsort, yolo_bytetrack, rtdetr_sort, yolo_tile_sort
python run_baselines_per_video.py \
    --baseline yolo_tile_sort \
    --data_dir data/SMOT4SB/phase2 \
    --num_videos 20
```

### Evaluate Custom Tracker

```python
# In Python script:
from utils.evaluation import evaluate_tracker
from baselines.yolo_tile_sort import YOLOTileSort

tracker = YOLOTileSort(
    yolo_model='yolov12s',
    tile_size=512,
    tile_overlap=128,
    conf_threshold=0.005
)

results = evaluate_tracker(
    tracker=tracker,
    data_dir='data/SMOT4SB/phase2',
    num_videos=10,
    output_dir='results/'
)

print(f"MOTA: {results['mota']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"FPS: {results['fps']:.2f}")
```

---

## Method Details

### 1. Confidence Threshold Calibration

**Standard YOLO**: confidence threshold = 0.5 (optimized for COCO)  
**Our approach**: confidence threshold = 0.005 (100× more permissive for small objects)

**Rationale**: At extreme scales, confidence scores are not calibrated for SMOT4SB's distribution. Extreme permissiveness recovers true positives at the cost of false positives—unavoidable without learned discrimination.

### 2. Motion Consistency Filtering

Use MOG2 background subtraction as secondary filter:

```python
# Keep YOLO detections with ≥30% overlap with motion mask
D_filtered = {d in D_yolo : IoU(d, R_motion) > 0.3}
```

**Note**: Motion filtering improves precision but reduces recall. Used as optional secondary filter, not primary detector.

### 3. Spatial Tiling (Primary Innovation)

Process image as overlapping tiles to increase relative object size:

```python
# Original: 44.8 px bird in 1920 px width = 2.3%
# Tiled:    44.8 px bird in 512 px tile = 8.8%

tile_size = 512
overlap = 128  # 25% overlap for tracking continuity

# Process tiles, merge detections via NMS
```

**Why it works**: Birds occupy larger proportion of tile receptive field, improving detection quality.  
**Trade-off**: Edge discontinuities may cause tracking fragmentation, but temporal continuity compensates.

### 4. Adaptive Tracking Parameters

| Parameter | Standard | Ours | Rationale |
|-----------|----------|------|-----------|
| `max_age` | 1 | 10 | Small objects frequently missed; longer memory needed |
| `min_hits` | 3 | 1 | Confirm tracks immediately; let pruning handle FP |
| `iou_threshold` | 0.5 | 0.01 | Fast-moving birds have <0.1 IoU between frames |

---

## Explored but Rejected Approaches

### ❌ Motion-Only Detection (MOG2 + SORT)
- **Result**: 0.1% precision, 21.2% recall, -793.2 MOTA
- **Reason**: Camera pan-tilt motion triggers foreground everywhere
- **Lesson**: Motion detection alone is infeasible without camera stabilization

### ❌ Multi-Detector Ensemble
- **Approach**: Weighted fusion of YOLO + MOG2 + Optical Flow
- **Result**: -13.2 MOTA after optimization (vs -1.49 for tiling)
- **Reason**: Optical flow adds 5+ sec/frame; MOG2 precision issues contaminate ensemble
- **Lesson**: Complex fusion doesn't outperform simple tiling. Requires more fine-tuning

### ❌ End-to-End Tracking (CenterTrack, FairMOT)
- **Reason**: Require GPU + extensive setup; no improvement over YOLO+SORT
- **Lesson**: Distribution mismatch affects all COCO-trained methods

### ❌ Appearance Features (DINO, CLIP)
- **Reason**: <1% improvement for 10× computational cost; birds too small for meaningful features
- **Lesson**: Low resolution makes appearance features unhelpful

### ❌ Optical Flow (RAFT)
- **Reason**: 5+ sec/frame overhead; no clear benefit on CPU
- **Lesson**: Motion prediction beyond Kalman filters not practical under constraints

**Conclusion**: Under CPU constraints, simplicity wins. Complex feature-rich methods consistently underperform spatial tiling.

---

## Evaluation Metrics

All metrics computed with IoU threshold τ = 0.1 (relaxed from standard 0.5, appropriate for tiny objects):

- **Precision**: TP / (TP + FP) — fraction of detections that are correct
- **Recall**: TP / (TP + FN) — fraction of ground truth detected
- **MOTA**: Multi-Object Tracking Accuracy — penalizes FN, FP, identity switches
- **HOTA**: Higher Order Tracking Accuracy — combines detection quality and association quality
- **FPS**: Frames per second (CPU-only inference)

---

## Understanding the Results

### Why is MOTA Negative?

MOTA < 0 means false positives + false negatives exceed ground truth objects. This is **expected** when applying general-purpose detectors to extreme scales without domain fine-tuning.

For SMOT4SB:
- Standard YOLO+SORT: -20.1 MOTA (20× more errors than objects)
- Tiling: -1.49 MOTA (1.5× more errors than objects)

Negative MOTA is not failure—it's expected behavior for distribution mismatch. Our tiling approach reduces errors by 18.6 MOTA points through relative object size scaling.

### Why Not Fine-Tune?

The assignment prohibits training/adapting deep learning model weights. Our approach stays within constraints by:
- Using off-the-shelf YOLO, RT-DETR (no fine-tuning)
- Adapting parameters (confidence threshold, tracking parameters)
- Exploring preprocessing (tiling, motion filtering)
- Optimizing fusion weights (hyperparameter tuning, not model training)

Production-grade performance would require domain fine-tuning with small-object augmentation, which is beyond assignment scope.

---

## Future Work

### Immediate Extensions (on current approach)
1. **Adaptive tiling**: Learn tile size from image content
2. **Tile boundary handling**: Soft blending of edge detections
3. **Temporal tiling**: Leverage motion across frames
4. **Hierarchical tiling**: Multi-level tile sizes for efficiency

### Production Improvements
1. **Domain fine-tuning**: Fine-tune YOLO on SMOT4SB with small-object augmentation
2. **Specialized architectures**: Cascade R-CNN, attention-refined networks
3. **Synthetic data**: Copy-paste tiny birds on real backgrounds
4. **Alternative modalities**: Thermal/multi-spectral imaging
5. **GPU acceleration**: Deploy tiling on GPU for 30-60 FPS

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{placeholder,
  title={YOLO-DA and YOLO-DAST: Domain-Aware and Spatial Tiling Approaches for Small Multi-Object Tracking},
  author={{placeholder}},
  year={2025},
  howpublished={\url{https://placeholder}}
}
```

---

## References

Key papers referenced in this work:

- **SORT**: Bewley et al. (2016) - Simple Online and Realtime Tracking
- **OC-SORT**: Cao et al. (2023) - Observation-centric SORT
- **ByteTrack**: Zhang et al. (2022) - Multi-object Tracking by Associating Every Detection Box
- **YOLO**: Jocher et al. (2023) - YOLOv8-v12
- **RT-DETR**: Lv et al. (2023) - DETRs Beat YOLOs on Real-time Object Detection
- **MOG2**: Zivkovic (2004) - Improved Adaptive Gaussian Mixture Model for Background Subtraction
- **HOTA**: Luiten et al. (2021) - HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking

See `references.bib` for complete bibliography.

AI tools have been used to write whole or part of this README.md.

---

## Troubleshooting

### Issue: Process hanging with motion_sort

**Symptom**: MOG2 produces 500+ detections/frame, Hungarian algorithm takes forever

**Solution**: Skip motion-only approaches. They fail due to camera motion triggering foreground everywhere.

```python
# In run_baselines_per_video.py, exclude:
# 'motion_sort'
# 'motion_yolo_sort'
```

### Issue: YOLO inference slow on CPU

**Solution**: Use smaller YOLO variants or reduce input size:

```python
# Slower: YOLOv12s with imgsz=1920
# Faster: YOLOv8s with imgsz=640 (but lower accuracy)
```

### Issue: Out of memory

**Solution**: Process fewer videos or reduce tile size:

```bash
python run_baselines_per_video.py --num_videos 10  # Process 10 instead of 96
```

---

## Contact & Support

For questions about the code, results, or methodology, refer to the paper: "YOLO-DA and YOLO-DAST: Domain-Aware and Spatial Tiling Approaches for Small Multi-Object Tracking"

---

## License

License: Academic Use Only

This code is provided for research and educational purposes.
If you use it, please cite the paper.
If you do not use it, that is also fine — the birds will continue flying regardless.
