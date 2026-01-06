import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    ["yolo_sort_coco", 11.8, 1532, 0.392, 0.055, -0.02, 0.016, 1000, 17, 0, 0.16],
    ["yolo_sort_tune", 8.96, 1591, 0.171, 0.233, -0.95, 0.0625, 1000, 123, 0.47, 0.27],
    ["yolo_bytetrack", 1.24, 1277, 0.336, 0.221, -1.57, 0.086, 1000, 16, 0.42, 0.22],
    ["yolo_sort", 1.216, 1590, 0.239, 0.243, -1.71, 0.0618, 373, 129, 0.5, 0.26],
    ["yolo_ocsort", 1.238, 1007, 0.239, 0.243, -1.71, 0.0616, 373, 128, 0.5, 0.26],
    ["rtdetr_sort", 1.29, 911, 0.284, 0.293, -1.32, 0.042, 980, 16, 0.52, 0.25],
    ["motion_sort", 15.9, 1748, 0.001, 0.70, -243, 0.0015, 1000, np.nan, np.nan, 0.36],
    ["dino_sort", 34.13, 1495, 0.001, 0.80, -81, 0.001, 1000, np.nan, np.nan, 0.37],
    ["yolo_tile_bytetrack", 2.48, 1503, 0.264, 0.372, -0.58, 0.112, 299, 59, 0, 0.33],
    ["rtdetr_tile_bytetrack", 1.08, 968, 0.01, 0.90, -206, 0.125, 1000, 17, 0.6, 0.5],
    ["ensemble_tracker", 2.71, 1985, 0.01, 0.70, -466, np.nan, 1000, np.nan, np.nan, 0.5],
    ["motion_yolo_sort", 5.6, 1600, 0.008, 0.122, -143, 0.005, 1000, np.nan, np.nan, 0.22],
]


cols = ["name", "fps", "mem_mb", "precision", "recall",
        "mota", "hota", "dotd", "id_switches", "motp", "iou"]

df = pd.DataFrame(data, columns=cols)

data = [
    # name, label, fps, precision, recall, f1, hota
    ("yolo_sort_coco", "YOLO+SORT (COCO)", 11.8, 0.392, 0.055, 0.096, 0.016),
    ("yolo_sort_tune", "YOLO-DA", 8.96, 0.171, 0.233, 0.197, 0.0625),
    ("yolo_sort", "YOLO-DA (res)", 1.216, 0.239, 0.243, 0.241, 0.0618),
    ("rtdetr_sort", "RT-DETR", 1.29, 0.284, 0.293, 0.288, 0.042),
    ("motion_sort", "MOG2", 15.9, 0.001, 0.70, 0.002, 0.0015),
    ("dino_sort", "DINO", 34.13, 0.001, 0.80, 0.002, 0.001),
    ("yolo_tile_bytetrack", "YOLO-DAST", 2.48, 0.264, 0.372, 0.308, 0.112),
    ("rtdetr_tile_bytetrack", "RTDETR-DAST", 1.08, 0.01, 0.90, 0.018, 0.125),
    ("ensemble_tracker", "Ensemble", 2.71, 0.01, 0.70, 0.02, np.nan),
    ("motion_yolo_sort", "MOG2 + YOLO", 5.6, 0.008, 0.122, 0.015, 0.005),
]

df = pd.DataFrame(
    data,
    columns=["id", "label", "fps", "precision", "recall", "f1", "hota"]
)


#################################
## FPS v HOTA
## Pereto line
#################################
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

# Scatter all points
for _, r in df.dropna(subset=["hota"]).iterrows():
    plt.scatter(r["fps"], r["hota"], s=80, color="tab:blue")
    plt.text(r["fps"] * 1.05, r["hota"] + 0.002, r["label"], fontsize=9)

# ---- Compute Pareto frontier ----
pts = df.dropna(subset=["hota"])[["fps", "hota"]].values
pts = pts[pts[:, 0].argsort()]  # sort by FPS

frontier = []
max_hota = -np.inf
for fps, hota in pts[::-1]:  # iterate from slowest → fastest
    if hota > max_hota:
        frontier.append((fps, hota))
        max_hota = hota

frontier = np.array(frontier)

# Plot frontier
plt.plot(
    frontier[:, 0],
    frontier[:, 1],
    color="black",
    linewidth=2,
    label="Pareto frontier"
)

# Highlight YOUR method
row = df[df["id"] == "yolo_tile_bytetrack"].iloc[0]
plt.scatter(row["fps"], row["hota"],
            s=300, marker="*", color="crimson", edgecolor="k", zorder=5)
plt.annotate(
    "YOLO-DAST\n(on Pareto frontier)",
    (row["fps"], row["hota"]),
    xytext=(row["fps"] * 2, row["hota"] - 0.015),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

plt.xscale("log")
plt.xlabel("FPS (log scale)")
plt.ylabel("HOTA")
plt.title("HOTA vs FPS Trade-off (Pareto Frontier)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#################################
## Precision v Recall
#################################
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))

# ---- Scatter points ----
ax.scatter(
    df["recall"],
    df["precision"],
    s=df["fps"] * 18,          # size ~ speed
    alpha=0.7,
    edgecolor="k",
    color="tab:blue"
)

# ---- Label points (except highlighted one) ----
for _, r in df.iterrows():
    if r["id"] == "yolo_tile_bytetrack":
        continue
    ax.text(
        r["recall"] + 0.005,
        r["precision"] + 0.005,
        r["label"],
        fontsize=8
    )

# ---- F1 iso-curves ----
recall = np.linspace(0.001, 1.0, 500)
for f1 in [0.05, 0.1, 0.2, 0.3]:
    precision = (f1 * recall) / (2 * recall - f1)
    precision[precision < 0] = np.nan
    ax.plot(recall, precision, "--", color="gray", alpha=0.5)
    ax.text(
        0.88,
        (f1 * 0.88) / (2 * 0.88 - f1),
        f"F1={f1}",
        fontsize=8,
        color="gray"
    )

# ---- Highlight YOUR method ----
row = df[df["id"] == "yolo_tile_bytetrack"].iloc[0]

ax.scatter(
    row["recall"],
    row["precision"],
    s=350,
    marker="*",
    color="crimson",
    edgecolor="k",
    zorder=6
)

ax.annotate(
    "YOLO-DAST\n(best PR balance)",
    xy=(row["recall"], row["precision"]),
    xytext=(row["recall"] - 0.28, row["precision"] + 0.18),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10,
    fontweight="bold"
)

# ---- Axes & formatting ----
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Precision–Recall Trade-off with F1 Iso-curves")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#################################
## detection bottleneck: recall vs hota
#################################
plt.figure(figsize=(7, 6))

for _, r in df.dropna(subset=["hota"]).iterrows():
    plt.scatter(r["recall"], r["hota"], s=80)
    plt.text(r["recall"] + 0.005, r["hota"] + 0.002, r["label"], fontsize=9)

plt.xlabel("Recall")
plt.ylabel("HOTA")
plt.title("Recall vs HOTA (Detection Bottleneck)")
plt.grid(True)
plt.tight_layout()
plt.show()


#################################
## F1 vs FPS
#################################
plt.figure(figsize=(7, 6))

for _, r in df.iterrows():
    plt.scatter(r["fps"], r["f1"], s=80)
    plt.text(r["fps"] * 1.02, r["f1"] + 0.005, r["label"], fontsize=9)

plt.xscale("log")
plt.xlabel("FPS (log scale)")
plt.ylabel("F1 score")
plt.title("F1 vs FPS")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


#################################
## stability: ID Switches
#################################
plt.figure(figsize=(9,4))
plt.bar(df["name"], df["id_switches"])
plt.yscale("log")
plt.ylabel("ID Switches (log)")
plt.title("Tracking Stability")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


#################################
## Cost analysis
#################################
fig, ax1 = plt.subplots(figsize=(10,5))

# Bar plot: FPS
ax1.bar(range(len(df)), df["fps"], color="tab:blue", alpha=0.7)
ax1.set_ylabel("FPS", color="tab:blue")

# Line plot: Memory
ax2 = ax1.twinx()
ax2.plot(range(len(df)), df["mem_mb"], color="tab:red", marker="o")
ax2.set_ylabel("Memory (MB)", color="tab:red")

# X-axis labels (THIS is the key fix)
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df["name"], rotation=45, ha="right")

plt.title("Runtime Cost on CPU")
plt.tight_layout()
plt.show()
