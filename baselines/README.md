# Baseline Implementations Summary

## Implemented Baselines

### 1. YOLO + SORT ✅
**Status:** Fully implemented  
**Config:** `configs/yolo_sort.yaml`  
**File:** `baselines/yolo_sort.py`

**Description:** Standard YOLO object detection + SORT tracking

**Variants:**
- yolo12n (nano - fastest)
- yolo12s (small - balanced)
- yolo12m (medium - more accurate)

**Run:**
```bash
python run_baseline.py --config configs/yolo_sort.yaml --max_videos 5
```

---

### 2. RT-DETR + SORT ✅
**Status:** Fully implemented  
**Config:** `configs/rtdetr_sort.yaml`  
**File:** `baselines/rtdetr_sort.py`

**Description:** RT-DETR (Refined DETR) transformer-based detector + SORT

**Variants:**
- rtdetr-l (large)
- rtdetr-x (extra large)

**Run:**
```bash
python run_baseline.py --config configs/rtdetr_sort.yaml --max_videos 5
```

---

### 3. CLIP + SORT ✅
**Status:** Fully implemented (but SLOW)  
**Config:** `configs/clip_sort.yaml`  
**File:** `baselines/clip_sort.py`

**Description:** CLIP zero-shot detection via sliding window + SORT

**Features:**
- Text-based detection: "a bird"
- Multi-scale sliding windows
- No training required

**⚠️ WARNING:** Very slow on CPU (sliding window approach)

**Run:**
```bash
python run_baseline.py --config configs/clip_sort.yaml --max_videos 1
```

---

### 4. DINO + SORT ✅
**Status:** Fully implemented  
**Config:** `configs/dino_sort.yaml`  
**File:** `baselines/dino_sort.py`

**Description:** Background subtraction for detection + DINOv2 features for appearance-based matching

**Features:**
- Background subtraction (MOG2)
- DINOv2 feature extraction
- Appearance-based re-identification

**Best for:** Videos with static/semi-static camera

**Run:**
```bash
python run_baseline.py --config configs/dino_sort.yaml --max_videos 5
```

---

### 5. CenterTrack ⚠️
**Status:** Placeholder only  
**Config:** `configs/centertrack.yaml`  
**File:** `baselines/centertrack.py`

**Description:** End-to-end detection + tracking

**⚠️ NOTE:** Requires official CenterTrack implementation:
https://github.com/xingyizhou/CenterTrack

---

## TODO Baselines

### 6. FairMOT ⏳
**Status:** Not implemented  
**Description:** End-to-end multi-object tracker

**Installation:** Requires fairMOT from https://github.com/ifzhang/FairMOT

### 7. MOTRv2 ⏳
**Status:** Not implemented  
**Description:** Transformer-based end-to-end tracker

**Installation:** Very heavy, may not be worth it on CPU

---

## Quick Start Guide

### Run Single Baseline
```bash
# YOLO (recommended to start)
python run_baseline.py --config configs/yolo_sort.yaml --max_videos 2

# rt-DETR
python run_baseline.py --config configs/rtdetr_sort.yaml --max_videos 2

# DINO
python run_baseline.py --config configs/dino_sort.yaml --max_videos 2
```

### Run All Baselines
```bash
bash run_all_experiments.sh
```

### Override Config Values
```bash
# Lower confidence threshold
python run_baseline.py --config configs/yolo_sort.yaml \
    --set detector.conf_threshold=0.05

# Detect all classes (debugging)
python run_baseline.py --config configs/yolo_sort.yaml \
    --set detector.params.detect_all_classes=true
```

---

## Baseline Comparison

| Baseline       | Speed | Accuracy (Expected) | CPU-Friendly | Notes |
|----------------|-------|---------------------|--------------|-------|
| YOLO12n + SORT | ⚡⚡⚡ Fast | ⭐⭐ Medium | ✅ Yes | Best starting point |
| YOLO12s + SORT | ⚡⚡ Medium | ⭐⭐⭐ Good | ✅ Yes | Balanced |
| RT-DETR + SORT | ⚡ Slow | ⭐⭐⭐ Good | ⚠️ OK | Transformer-based |
| CLIP + SORT    | 🐌 Very Slow | ⭐ Low | ❌ No | Zero-shot, for exploration |
| DINO + SORT    | ⚡⚡ Medium | ⭐⭐ Medium | ✅ Yes | Good for static camera |
| CenterTrack    | ? | ⭐⭐⭐ Good | ? | Not implemented |

---

## Installation Requirements

### Core (required for all)
```bash
pip install numpy opencv-python scipy filterpy motmetrics ultralytics torch torchvision tqdm pyyaml
```

### For CLIP baseline
```bash
pip install git+https://github.com/openai/CLIP.git
```

### For DINO baseline
```bash
# Uses torch.hub, no extra install needed
```

### For CenterTrack/FairMOT
See their official repos for installation instructions.

---

## Results Location

All results saved to:
```
results/baselines/
├── yolo12n_sort/
│   ├── video_1/
│   │   ├── video_1_predictions.txt
│   │   └── video_1_metrics.json
│   └── summary.json
├── rtdetr_sort/
└── ...
```

---

## Tips for Small Bird Detection

1. **Lower confidence threshold**: Try 0.05 or 0.01
   ```bash
   --set detector.conf_threshold=0.05
   ```

2. **Test on few videos first**: Use `--max_videos 2`

3. **Check debug output**: Enable verbose mode in config

4. **YOLO may not work well**: Birds are too small for COCO-trained YOLO
   - That's why you need your novel approach!

---

## Adding New Baselines

1. Create `baselines/my_tracker.py` inheriting from `BaseTracker`
2. Implement 3 methods: `_initialize_detector`, `_initialize_tracker`, `_detect_frame`
3. Create `configs/my_tracker.yaml`
4. Register in `run_baseline.py` → `load_tracker()`
5. Run: `python run_baseline.py --config configs/my_tracker.yaml`

See `baselines/yolo_sort.py` as a template!
