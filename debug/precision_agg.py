import numpy as np
import pandas as pd

precisions = [0.5496, 0.2915, 0.9682, np.nan, 0.28, np.nan, np.nan, 0.7863, 0.5097, 0.0, np.nan, np.nan, np.nan, np.nan, 0.6354, 0.6579, 0.6479, 0.5350, 0.0, 0.5906, 0.1236, 0.0, np.nan, 0.2223, 0.0, 0.0909, 0.5152, 0.3497, np.nan, 0.0, 0.0037, np.nan]
fps = [6662, 1356, 5, 0, 90, 0, 0, 677, 1316, 14, 0, 0, 0, 0, 4450, 5679, 300, 292, 143, 174, 3603, 45, 0, 1581, 23, 20, 256, 292, 0, 136, 40347, 0]

total_tp = 0
total_fp = 0

for prec, fp in zip(precisions, fps):
    if pd.isna(prec) or fp == 0:
        continue

    if prec == 0:
        tp = 0
    elif prec == 1:
        tp = fp  # Edge case
    else:
        tp = fp * (prec / (1 - prec))  # TP = FP * (P/(1-P))

    total_tp += tp
    total_fp += fp

overall_precision = total_tp / (total_tp + total_fp)
print(f"Overall precision: {overall_precision:.3f}")
