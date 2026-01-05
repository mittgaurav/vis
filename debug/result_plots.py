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


#################################
## FPS v HOTA
#################################
plt.figure(figsize=(6,5))
plt.scatter(df["fps"], df["hota"], s=80)

for i, name in enumerate(df["name"]):
    plt.text(df["fps"][i]*1.05, df["hota"][i], name, fontsize=8)

plt.xscale("log")
plt.xlabel("FPS (log scale)")
plt.ylabel("HOTA")
plt.title("Accuracy–Speed Trade-off")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#################################
## Precision v Recall
#################################
import numpy as np
import matplotlib.pyplot as plt

# Assume df has numeric precision, recall in [0,1]
fig, ax = plt.subplots(figsize=(7,6))

# Scatter
scatter = ax.scatter(
    df["recall"],
    df["precision"],
    s=df["fps"] * 20,
    alpha=0.75,
    edgecolor="k"
)

# Label points
for i, name in enumerate(df["name"]):
    ax.text(df["recall"][i] + 0.004,
            df["precision"][i] + 0.004,
            name,
            fontsize=8)

# ---- F1 iso-curves ----
r = np.linspace(0.001, 1.0, 500)
for f1 in [0.05, 0.1, 0.2, 0.3]:
    p = (f1 * r) / (2*r - f1)
    p[p < 0] = np.nan
    ax.plot(r, p, "--", color="gray", alpha=0.6)
    ax.text(0.85, (f1*0.85)/(2*0.85-f1),
            f"F1={f1}",
            fontsize=8, color="gray")

# Highlight YOUR method
row = df[df["name"] == "yolo_tile_bytetrack"].iloc[0]
ax.scatter(row["recall"], row["precision"],
           s=300, marker="*", color="red", edgecolor="k", zorder=5)
ax.annotate(
    "Only method with\nusable Precision & Recall",
    (row["recall"], row["precision"]),
    xytext=(row["recall"]-0.25, row["precision"]+0.15),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Precision–Recall Trade-off with F1 Iso-curves")
ax.grid(True, alpha=0.3)

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
