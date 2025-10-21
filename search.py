import pandas as pd
import numpy as np
from stable_features import build_tick_features
from utils import load_any
import numpy as np

# Load data (assumes y is present)
askR, bidR, askS, bidS, askN, bidN, y = load_any('train.csv.gz', L_expected=8, has_y=True)



def build_candidate_features(askR, bidR, askS, bidS, askN, bidN, use_levels=4):
    Sa = np.nan_to_num(askS[:, :use_levels], copy=False)
    Sb = np.nan_to_num(bidS[:, :use_levels], copy=False)
    A = np.nan_to_num(askR[:, :use_levels], copy=False)
    B = np.nan_to_num(bidR[:, :use_levels], copy=False)

    # 1. Multi-level imbalance (mean over top 4)
    multi_level_imb = np.mean((Sb - Sa) / (Sb + Sa + 1e-8), axis=1)
    # 2. Depth ratio (sum bid / sum ask)
    cum_bid = Sb.sum(axis=1)
    cum_ask = Sa.sum(axis=1)
    depth_ratio = cum_bid / (cum_ask + 1e-8)
    # 3. Slope diff (slope_a - slope_b)
    def slope(P, S):
        cs = np.cumsum(S, axis=1)
        x = cs - cs.mean(axis=1, keepdims=True)
        y = P - P.mean(axis=1, keepdims=True)
        num = (x * y).sum(1)
        den = (x * x).sum(1)
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0))
    slope_a = slope(A, Sa)
    slope_b = slope(B, Sb)
    slope_diff = slope_a - slope_b
    # 4. Top-of-book pressure (bid0/ask0)
    top_book_pressure = Sb[:,0] / (Sa[:,0] + 1e-8)
    # 5. Level 2 imbalance
    if Sb.shape[1] > 2 and Sa.shape[1] > 2:
        lvl2_imb = (Sb[:,2] - Sa[:,2]) / (Sb[:,2] + Sa[:,2] + 1e-8)
    else:
        lvl2_imb = np.zeros(Sb.shape[0])
    return np.column_stack([
        multi_level_imb,
        depth_ratio,
        slope_diff,
        top_book_pressure,
        lvl2_imb
    ])

candidate_names = [
    "multi_level_imb",
    "depth_ratio",
    "slope_diff",
    "top_book_pressure",
    "lvl2_imb"
]


fractions = [0.4, 0.6, 0.8, 1.0]
stable_feature_names = [
    "qi_level0",              # 0: Queue imbalance at level 0
    "qi_level1",              # 1: Queue imbalance at level 1
    "top_book_pressure",      # 2: Top-of-book pressure
    "top_book_pressure_l1",   # 3: Level 1 pressure
    "volume_imb",             # 4: Total volume imbalance
    "slope_a",                # 5: Ask book slope
    "slope_b",                # 6: Bid book slope
    "ofi_proxy",              # 7: Order Flow Imbalance proxy
    "microprice",             # 8: Size-weighted mid price
    "multi_level_imb",        # 9: Mean imbalance across levels
    "depth_ratio",            # 10: Total bid/ask ratio
    "relative_spread",        # 11: Normalized spread
    "qi_change",              # 12: Change in imbalance
    "order_size_ratio"        # 13: Average order size ratio
]

for frac in fractions:
    N = int(frac * askR.shape[0])
    askR_f, bidR_f, askS_f, bidS_f, askN_f, bidN_f, y_f = askR[:N], bidR[:N], askS[:N], bidS[:N], askN[:N], bidN[:N], y[:N]
    X_f = build_tick_features(askR_f, bidR_f, askS_f, bidS_f, askN_f, bidN_f, use_levels=4)
    cand_f = build_candidate_features(askR_f, bidR_f, askS_f, bidS_f, askN_f, bidN_f, use_levels=4)
    label = f"{int(frac*100)}% of data" if frac < 1.0 else "all data"
    print(f"\nCorrelation between y and each stable feature ({label}):")
    for i in range(X_f.shape[1]):
        name = stable_feature_names[i] if i < len(stable_feature_names) else f"stable_{i+1}"
        corr = np.corrcoef(X_f[:, i], y_f)[0, 1]
        print(f"{name}: {corr:.5f}")
    print(f"Correlation between y and candidate features ({label}):")
    for i in range(cand_f.shape[1]):
        name = candidate_names[i]
        corr = np.corrcoef(cand_f[:, i], y_f)[0, 1]
        print(f"{name}: {corr:.5f}")

    # Print summary statistics for y
    print(f"\nSummary statistics for y ({label}):")
    print(f"  mean: {np.mean(y_f):.5f}, std: {np.std(y_f):.5f}, min: {np.min(y_f):.5f}, max: {np.max(y_f):.5f}")
    print(f"Summary statistics for each stable feature ({label}):")
    for i in range(X_f.shape[1]):
        vals = X_f[:, i]
        print(f"  stable_{i+1}: mean={np.mean(vals):.5f}, std={np.std(vals):.5f}, min={np.min(vals):.5f}, max={np.max(vals):.5f}")