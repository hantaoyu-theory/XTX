import numpy as np

# ===== Basic helpers =====

def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

# ===== Per-tick features (levels 1..use_levels) =====

def build_tick_features(askRate, bidRate, askSize, bidSize, askNc, bidNc, use_levels: int = 4):
    """
    Inputs are arrays of shape (T, L). We clip to the first `use_levels`.
    Prices NaNs -> 0 (rare in top levels); sizes/NC NaNs -> 0 (empty book beyond that level).
    Returns X_now with shape (T, F_now).
    """
    L = use_levels
    A = np.nan_to_num(askRate[:, :L], copy=False)
    B = np.nan_to_num(bidRate[:, :L], copy=False)
    Sa = np.nan_to_num(askSize[:, :L], copy=False)
    Sb = np.nan_to_num(bidSize[:, :L], copy=False)
    Na = np.nan_to_num(askNc[:, :L],   copy=False)
    Nb = np.nan_to_num(bidNc[:, :L],   copy=False)

    best_a = A[:, 0]                          # best ask price
    best_b = B[:, 0]                          # best bid price
    spread = best_a - best_b                  # spread
    mid = 0.5 * (best_a + best_b)             # mid-price

    top_sz_a = Sa[:, 0]
    top_sz_b = Sb[:, 0]
    microprice = _safe_div(best_a * top_sz_b + best_b * top_sz_a, top_sz_a + top_sz_b)
    qi = _safe_div(top_sz_b - top_sz_a, top_sz_a + top_sz_b)  # queue imbalance (top level)

    dep_a = Sa.sum(1)                         # total ask depth
    dep_b = Sb.sum(1)                         # total bid depth

    def slope(P, S):
        # simple price vs cumulative-size regression slope per tick (proxy for book slope)
        cs = np.cumsum(S, axis=1)
        x = cs - cs.mean(axis=1, keepdims=True)
        y = P - P.mean(axis=1, keepdims=True)
        num = (x * y).sum(1)
        den = (x * x).sum(1)
        return _safe_div(num, den)

    slope_a = slope(A, Sa)
    slope_b = slope(B, Sb)

    # Side-level imbalances (per level 1..L)
    lvl_imb = _safe_div(Sb - Sa, (Sb + Sa))   # (T, L)
    lvl_imb_flat = lvl_imb.reshape(lvl_imb.shape[0], -1)

    # Select only 10 features for now
    # Here are 10: best_a, best_b, spread, mid, microprice, qi, dep_a, dep_b, slope_a, slope_b
    X_now = np.stack([
        best_a, best_b, spread, mid, microprice,
        qi, dep_a, dep_b, slope_a, slope_b
    ], axis=1)
    return X_now  # (T, 10)

# ===== Strictly-causal temporal features =====

def add_causal_temporal_features(X, windows=(5, 20, 60)):
    """
    For each feature, add causal rolling means and one-lag differences.
    All operations ensure that feature at time t uses only <= t.
    Returns augmented array (T, F_aug).
    """
    T, F = X.shape
    feats = [X]

    # Only keep lag-1 as the single most important temporal feature
    lag1 = np.vstack([np.zeros((1, F), dtype=X.dtype), X[:-1]])
    feats.append(lag1)

    # Do not add EMAs or residuals for now

    return np.concatenate(feats, axis=1)  # (T, 2*F)
