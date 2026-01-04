# Configuration Files Guide

All baseline and experimental approaches are configured via YAML files in the `configs/` directory. This guide explains each configuration, what parameters do, and how to modify them for your experiments.

---

## Configuration Structure

Each config file follows this pattern:

```yaml
# Detector configuration
detector:
  type: <detector_type>
  model_name: <model_variant>
  conf_threshold: <float>  # Confidence threshold for detections
  imgsz: <int>             # Input size for YOLO/RT-DETR
  # ... detector-specific params

# Tracker configuration  
tracker:
  type: <tracker_type>
  max_age: <int>           # Frames to keep unmatched track
  min_hits: <int>          # Detections before confirming track
  iou_threshold: <float>   # IoU threshold for matching
  # ... tracker-specific params
```

---

## Running Baselines

### Example: `yolo_sort.yaml` - YOLO + SORT (Baseline)

**File:** `configs/yolo_sort.yaml`  
**Implementation:** `baselines/yolo_sort.py`  
**Status:** ✅ Fully tested  
**Results:** MOTA=-20.1, Precision=0.397, Recall=0.218, FPS=3.9

### Configuration

```yaml
detector:
  type: yolo
  model_name: yolov12s          # YOLO variant (yolov8s, yolov11s, yolov12s, etc.)
  conf_threshold: 0.005          # Lowered from standard 0.5 for small objects
  imgsz: 1920                    # Input size (1920×1920 for SMOT4SB Phase 2)
  nms_threshold: 0.5             # Non-maximum suppression threshold
  device: cpu                    # Use CPU (gpu for GPU)

tracker:
  type: sort
  max_age: 10                    # Keep track alive for 10 frames without detection
  min_hits: 1                    # Confirm track after 1 detection (not 3)
  iou_threshold: 0.01            # Very loose matching for fast-moving small objects
```

### Why These Parameters?

| Parameter | Value | Reason |
|-----------|-------|--------|
| `conf_threshold` | 0.005 | Standard 0.5 rejects true positives; 100× more permissive needed |
| `imgsz` | 1920 | Match SMOT4SB Phase 2 dimensions; preserve small object detail |
| `max_age` | 10 | Small objects frequently missed; longer memory reduces track fragmentation |
| `min_hits` | 1 | Confirm immediately; standard 3 wastes early detections at small scale |
| `iou_threshold` | 0.01 | Birds moving 30-50 px/frame have <0.1 IoU between frames; 0.5 breaks tracking |

### How to Run

```bash
# Run on 20 videos
python run_baselines_per_video.py --baseline yolo_sort --num_videos 20

# Run on all 96 training videos
python run_baselines_per_video.py --baseline yolo_sort --num_videos 96

# Override config values on command line
python run_baselines_per_video.py --baseline yolo_sort \
    --num_videos 20 \
    --set detector.conf_threshold=0.01 \
    --set tracker.max_age=20
```

### Tuning Guide

**To improve recall (detect more birds):**
```yaml
detector:
  conf_threshold: 0.001  # Even more permissive (more false positives)
```

**To improve precision (fewer false positives):**
```yaml
detector:
  conf_threshold: 0.01   # Stricter (fewer true positives)
  nms_threshold: 0.7     # Stricter NMS
```

**For faster inference (sacrifice accuracy):**
```yaml
detector:
  model_name: yolov8n    # Nano variant (2× faster)
```

**For slower but more accurate:**
```yaml
detector:
  model_name: yolov12m   # Medium variant (slower but better)
```
