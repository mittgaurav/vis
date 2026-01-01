"""
Motion-gated YOLO + DINO + Appearance-aware SORT
Designed for small bird tracking (SMOT)
"""

import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from baselines.base_tracker import BaseTracker
from trackers.sort_tracker import KalmanBoxTracker


class Track:
    def __init__(self, bbox, track_id, feature=None):
        self.kalman = KalmanBoxTracker(bbox)
        self.id = track_id
        self.feature = feature
        self.hits = 1
        self.age = 0
        self.time_since_update = 0


class MotionYOLODINOTracker(BaseTracker):

    def _initialize_detector(self):
        """Initialize all detection components"""
        cfg = self.config['detector']

        # Motion detection
        m = cfg['motion_detection']
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=m['history'],
            varThreshold=m['var_threshold'],
            detectShadows=False
        )
        self.motion_min_area = m['min_area']
        self.motion_max_area = m['max_area']

        # YOLO
        from ultralytics import YOLO
        y = cfg['yolo']
        self.yolo = YOLO(f"{y['model_name']}.pt").to(self.device)
        self.yolo_conf = y['conf_threshold']
        self.filter_class = y.get('filter_class', 14)
        self.yolo_min_area = y.get('min_area', 0)
        self.yolo_max_area = y.get('max_area', np.inf)

        # DINO
        d = cfg['dino']
        self.dino = torch.hub.load('facebookresearch/dinov2', d['model_name'])
        self.dino.eval().to(self.device)
        self.dino_similarity_threshold = d.get('similarity_threshold', 0.7)

        return self

    def _initialize_tracker(self):
        """Initialize tracker parameters"""
        p = self.config['tracker']['params']
        self.max_age = p['max_age']
        self.min_hits = p['min_hits']
        self.iou_thr = p['iou_threshold']
        self.app_w = p.get('appearance_weight', 0.5)
        self.iou_w = p.get('iou_weight', 0.5)

        self.tracks = []
        self.next_id = 1

        return self

    # ---------------- RESET ----------------
    def reset(self):
        """Reset tracker state and background model"""
        self.tracks = []
        self.next_id = 1
        self._features = None
        if hasattr(self, 'bg'):
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=self.config['detector']['motion_detection']['history'],
                varThreshold=self.config['detector']['motion_detection']['var_threshold'],
                detectShadows=False
            )

    # ---------------- MOTION ----------------
    def _motion_regions(self, img):
        fg = self.bg.apply(img)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3)))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for c in cnts:
            a = cv2.contourArea(c)
            if self.motion_min_area < a < self.motion_max_area:
                x, y, w, h = cv2.boundingRect(c)
                regions.append((x, y, w, h))
        return regions

    # ---------------- YOLO ----------------
    def _yolo_detect(self, img, regions):
        dets = []

        if not regions:
            regions = [(0, 0, img.shape[1], img.shape[0])]

        for x, y, w, h in regions:
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            results = self.yolo(crop, conf=self.yolo_conf, device=self.device, verbose=False)
            for r in results:
                for b in r.boxes:
                    if int(b.cls[0]) != self.filter_class:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    w2, h2 = x2 - x1, y2 - y1
                    area = w2 * h2
                    if area < self.yolo_min_area or area > self.yolo_max_area:
                        continue

                    dets.append([x+x1, y+y1, w2, h2, float(b.conf[0])])

        return np.array(dets) if dets else np.empty((0, 5))

    # ---------------- DINO ----------------
    def _extract_features(self, img, dets):
        feats = []
        for d in dets:
            x, y, w, h = map(int, d[:4])
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                feats.append(np.zeros(384))
                continue

            # Convert to RGB and resize with OpenCV (avoids MPS bicubic crash)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Convert to tensor
            t = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0)/255.
            t = t.to(self.device)

            # Extract features
            with torch.no_grad():
                f = self.dino(t).cpu().numpy().flatten()
                f = f / (np.linalg.norm(f) + 1e-6)
            feats.append(f)

        return np.array(feats)


    # ---------------- TRACKING ----------------
    def _associate(self, dets, feats):
        if not self.tracks:
            return [], list(range(len(dets))), []

        cost = np.zeros((len(dets), len(self.tracks)))
        for i, d in enumerate(dets):
            for j, t in enumerate(self.tracks):
                iou = self._iou(d[:4], t.kalman.get_state())
                app = 0.5
                if t.feature is not None:
                    app = 1 - np.dot(feats[i], t.feature)
                cost[i, j] = self.iou_w*(1-iou) + self.app_w*app

        r, c = linear_sum_assignment(cost)
        matches, ud, ut = [], [], []

        for i in range(len(dets)):
            if i not in r:
                ud.append(i)
        for j in range(len(self.tracks)):
            if j not in c:
                ut.append(j)

        for i, j in zip(r, c):
            if cost[i, j] < 0.5:
                matches.append((i, j))
            else:
                ud.append(i)
                ut.append(j)

        return matches, ud, ut

    def _iou(self, a, b):
        x1, y1, w1, h1 = a
        x2, y2, w2, h2 = b
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter = max(0, xi2-xi1)*max(0, yi2-yi1)
        return inter/(w1*h1+w2*h2-inter+1e-6)

    # ---------------- PIPELINE ----------------
    def _detect_frame(self, img):
        regions = self._motion_regions(img)
        dets = self._yolo_detect(img, regions)
        self._features = self._extract_features(img, dets) if len(dets) else np.empty((0, 384))
        return dets

    def _track_frame(self, dets):
        feats = self._features
        for t in self.tracks:
            t.time_since_update += 1

        matches, ud, ut = self._associate(dets, feats)

        for i, j in matches:
            self.tracks[j].kalman.update(dets[i][:4])
            self.tracks[j].feature = feats[i]
            self.tracks[j].hits += 1
            self.tracks[j].time_since_update = 0

        for i in ud:
            self.tracks.append(Track(dets[i][:4], self.next_id, feats[i]))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                out.append(np.concatenate([t.kalman.get_state(), [t.id]]))
        return np.array(out) if out else np.empty((0, 5))
