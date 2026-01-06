import json
import pandas as pd
import numpy as np

# Load JSON
with open("results/per_video_baseline/aggregate_results/aggregate_summary.json", "r") as f:
    data = json.load(f)


# Function to compute F1
def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


# Collect rows
rows = []
for key, val in data.items():
    metrics = val["avg_metrics"]
    stats = val["avg_stats"]

    precision = metrics.get("precision", np.nan)
    recall = metrics.get("recall", np.nan)
    f1 = compute_f1(precision, recall)

    row = {
        "Experiment": key,
        "fps": stats.get("avg_fps", np.nan),
        "Mem MB": stats.get("avg_memory_mb", np.nan),
        "Precision": round(precision * 100, 1),
        "Recall": round(recall * 100, 1),
        "F1": round(f1, 3),
        "mota": round(metrics.get("mota", np.nan), 3),
        "hota": round(metrics.get("hota", np.nan), 3),
        "Dotd": round(metrics.get("dotd", np.nan), 1),
        "ID Switches": round(metrics.get("num_switches", np.nan), 1),
        "motp": round(metrics.get("motp", np.nan), 2),
        "iou": round(metrics.get("mean_iou", np.nan), 2)
    }
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Optional: sort by fps or any metric
df = df.sort_values(by="Experiment")

# Print nicely
print(df.to_string(index=False))

# Save to CSV or LaTeX table
df.to_csv("summary_table.csv", index=False)

# Generate LaTeX tabular
latex_table = df.to_latex(index=False, caption="SMOT4SB tracking results per model", float_format="%.3f")
with open("summary_table.tex", "w") as f:
    f.write(latex_table)
