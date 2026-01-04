# Novel Tracking Approaches (Exploratory)

Your novel solutions to beat the baselines!

## Implemented Approaches

### 1. Motion-Guided Multi-Scale Tracker ⭐ (RECOMMENDED)
**File:** `motion_multiscale_tracker.py`  
**Config:** `configs/motion_multiscale_tracker.yaml`

**Pipeline:**
```
Image → Motion Detection → Multi-Scale YOLO → DINO Features → Enhanced Tracking
         (BG Subtract)      (1x, 1.5x, 2x)     (Re-ID)        (Kalman + Flow)
```

**Key Features:**
- **Stage 1:** Background subtraction finds moving regions
- **Stage 2:** Multi-scale YOLO runs only on motion regions (efficient!)
- **Stage 3:** DINO extracts appearance features for re-identification
- **Stage 4:** Optical flow enhances motion prediction
- **Tracking:** Custom tracker with appearance + motion fusion

**Why it should work:**
✅ Motion detection reduces false positives  
✅ Multi-scale handles birds at different distances  
✅ DINO helps with occlusions and re-identification  
✅ Optical flow improves prediction for fast-moving birds  
✅ Efficient - only processes motion regions

**Run:**
```bash
python run_baseline.py --config configs/motion_multiscale_tracker.yaml --max_videos 2 --device cuda
```


---

### 3. Multi-Detector Ensemble (MOST ROBUST)
**File:** `ensemble_tracker.py`  
**Config:** `configs/ensemble_tracker.yaml`

**Pipeline:**
```
Image → YOLO Detector ──┐
     → BG Subtractor ───┼→ Weighted Fusion → NMS → SORT Tracking
     → Optical Flow ────┘
```

**Key Features:**
- Runs 3 detectors in parallel: YOLO, Background Subtraction, Optical Flow
- Weighted voting/fusion of detections
- Confidence aggregation
- Standard SORT tracking

**Why it should work:**
✅ Redundancy - if one detector fails, others compensate  
✅ Voting increases confidence  
✅ Combines strengths of multiple methods  
✅ Robust to different scenarios

**Run:**
```bash
python run_baseline.py --config configs/ensemble_tracker.yaml --max_videos 2 --device cuda
```

---

## Comparison of Novel Approaches

| Approach | Complexity | Speed | Expected Performance | Best For |
|----------|-----------|-------|---------------------|----------|
| Motion-Guided Multi-Scale | High | Medium | ⭐⭐⭐⭐⭐ Excellent | All scenarios |
| RAFT-DINO | Medium | Fast | ⭐⭐⭐⭐ Very Good | Static camera |
| Ensemble | Medium | Slow | ⭐⭐⭐⭐ Very Good | Robust results |

---

## Implementation Status

✅ **Approach 1: Motion-Guided Multi-Scale** - Fully implemented  
⏳ **Approach 2: RAFT-DINO** - Skeleton created, needs completion  
⏳ **Approach 3: Ensemble** - Skeleton created, needs completion

---

## Quick Start

### Test Novel Approach
```bash
# Approach 1 (recommended)
python run_baseline.py --config configs/motion_multiscale_tracker.yaml --max_videos 2 --device cuda

# Compare with baseline
python run_baselines_per_video.py \
    --configs configs/yolo12n_sort.yaml configs/motion_multiscale_tracker.yaml \
    --max_videos 5
```

### Tune Parameters
```bash
# Lower motion detection threshold
python run_baseline.py --config configs/motion_multiscale_tracker.yaml \
    --set detector.motion_detection.min_area=15 \
    --max_videos 2

# Adjust YOLO confidence
python run_baseline.py --config configs/motion_multiscale_tracker.yaml \
    --set detector.yolo.conf_threshold=0.03 \
    --max_videos 2
```

---

## Expected Results

Based on the design, **Motion-Guided Multi-Scale Tracker** should:
- **Precision:** Higher than YOLO alone (motion filtering removes false positives)
- **Recall:** Higher than YOLO alone (multi-scale + low threshold catches small birds)
- **MOTA:** +10-20% improvement over best baseline
- **ID Switches:** Fewer (DINO features help re-identification)
- **Speed:** ~3-5 FPS on GPU (slower than YOLO alone, but much better accuracy)

---

## Key Innovations for Your Report

### Technical Contributions:
1. **Multi-Scale Detection on Motion Regions** - Novel combination reducing computation
2. **Appearance-Enhanced Tracking** - Using DINO for re-identification in SORT framework
3. **Hybrid Motion Prediction** - Combining Kalman filter with optical flow
4. **Motion-Guided ROI Selection** - Smart region proposal instead of full image

### Ablation Studies to Run:
- With/without motion detection (Stage 1)
- With/without multi-scale (Stage 2)
- With/without DINO features (Stage 3)
- With/without optical flow (Stage 4)

This shows which components contribute most!

---

## Debugging Tips

### No detections?
- Lower `detector.motion_detection.min_area` (try 10)
- Lower `detector.yolo.conf_threshold` (try 0.01)
- Check motion regions: `--set debug.verbose=true`

### Too many false positives?
- Increase `detector.motion_detection.var_threshold`
- Increase `detector.yolo.conf_threshold`
- Adjust `tracker.params.min_hits` (require more confirmations)

### Poor re-identification?
- Increase `tracker.params.appearance_weight`
- Lower `tracker.params.max_age` (don't keep tracks too long)

---

## Exploratory Approaches Investigated

We explored several novel approaches to improve small bird tracking:

1. **Motion-Guided Multi-Scale YOLO** 
   - Motivation: Run YOLO at multiple scales on motion regions to find small objects
   - Challenge: Running YOLO multiple times per frame (15+ sec/frame) made it impractical
   - Learning: Multi-scale inference adds complexity without sufficient speedup

2. **Tiled YOLO Detection**
   - Motivation: Process large image in tiles to magnify small objects
   - Result: Recall dropped to 9.6% (worse than baseline 23%)
   - Finding: Tile boundaries cause tracking discontinuities; full-image YOLO is more effective

3. **Motion + YOLO + DINO Features**
   - Motivation: Combine motion detection, YOLO, and appearance features
   - Challenge: DINO feature extraction too slow (10+ sec/frame)
   - Conclusion: Feature extraction overhead not justified for this dataset

4. **Optical Flow + ByteTrack**
   - Motivation: Use RAFT optical flow to predict track positions
   - Issue: RAFT integration problematic; Farneback flow insufficient
   - Insight: Motion prediction requires better flow estimates than available

5. **Motion-Filtered YOLO** (Best Exploratory)
   - Motivation: Fast alternative - run YOLO once, filter by motion regions
   - Speed: ~0.2s/frame (comparable to baseline)

## Summary of Findings

Despite extensive exploration, simple YOLO+SORT baseline proved most robust.
Motion-based detection achieved higher recall (33%) but at cost of precision (0.1%).
The fundamental limitation is YOLO's lack of specialization for tiny birds.
