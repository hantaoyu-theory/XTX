# utils.py
import os, re
import numpy as np
import pandas as pd

# -----------------------------
# Flexible data loader (CSV/CSV.GZ/NPZ)
# -----------------------------
# CSV/CSV.GZ expected columns (0-based ok):
#   askRate_0..7, bidRate_0..7, askSize_0..7, bidSize_0..7, askNc_0..7, bidNc_0..7, [y]
# Returns arrays shaped (T, L) for each field, plus y (T,) if present.

def _extract_levels(df, base, L, zero_based_ok=True):
    """
    Return a (T, L) array for a group like base='askRate'.
    Works with names like askRate_0.., askRate_1.., askRateL0/L1, askRate[0]/[1], askRate0/1.
    Auto-detects whether levels start at 0 or 1.
    """
    levels_seen = []
    patterns = [
        rf"^{base}[_\-]?(\d+)$",
        rf"^{base}L(\d+)$",
        rf"^{base}\[(\d+)\]$",
        rf"^{base}(\d+)$",
    ]
    for col in df.columns:
        s = str(col)
        for pat in patterns:
            m = re.match(pat, s)
            if m:
                try:
                    levels_seen.append(int(m.group(1)))
                except ValueError:
                    pass
    zero_based = (min(levels_seen) == 0) if levels_seen else zero_based_ok
    desired_levels = list(range(0, L)) if zero_based else list(range(1, L+1))

    level_to_col = {}
    for col in df.columns:
        s = str(col)
        for pat in patterns:
            m = re.match(pat, s)
            if m:
                lvl = int(m.group(1))
                if lvl in desired_levels and lvl not in level_to_col:
                    level_to_col[lvl] = col

    # Fallback: first L columns with the prefix
    if len(level_to_col) < L:
        prefixed = [c for c in df.columns if str(c).startswith(base)]
        if len(prefixed) >= L:
            for i in range(L):
                level_to_col[desired_levels[i]] = prefixed[i]

    cols = [level_to_col.get(lvl) for lvl in desired_levels]
    if any(c is None for c in cols):
        missing = [desired_levels[i] for i, c in enumerate(cols) if c is None]
        raise ValueError(f"Missing {base} levels {missing}. Found: {level_to_col}")
    return df[cols].to_numpy(dtype=float)

def load_any(path, L_expected=8, has_y=True):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        d = np.load(path)
        askRate = d['askRate']; bidRate = d['bidRate']
        askSize = d['askSize']; bidSize = d['bidSize']
        askNc   = d['askNc'];   bidNc   = d['bidNc']
        y = d['y'] if ('y' in d and has_y) else None
        return askRate, bidRate, askSize, bidSize, askNc, bidNc, y

    # CSV / CSV.GZ
    df = pd.read_csv(path, compression='infer')
    def infer_L(prefix):
        pref = [c for c in df.columns if str(c).startswith(prefix)]
        return min(len(pref), L_expected) if pref else L_expected
    L = max(infer_L('askRate'), infer_L('bidRate'),
            infer_L('askSize'), infer_L('bidSize'),
            infer_L('askNc'),   infer_L('bidNc'))

    askRate = _extract_levels(df, 'askRate', L)
    bidRate = _extract_levels(df, 'bidRate', L)
    askSize = _extract_levels(df, 'askSize', L)
    bidSize = _extract_levels(df, 'bidSize', L)
    askNc   = _extract_levels(df, 'askNc',   L)
    bidNc   = _extract_levels(df, 'bidNc',   L)
    y = df['y'].to_numpy(dtype=float) if ('y' in df.columns and has_y) else None
    return askRate, bidRate, askSize, bidSize, askNc, bidNc, y

# -----------------------------
# Sequencing + metric
# -----------------------------

def make_sequences(X, y, W):
    """
    X: (T, F), y: (T,)
    Returns:
      Xseq: (N, W, F) with windows ending at t = W-1..T-1
      yseq: (N,)     aligned to window end index (y_t)
    """
    T, F = X.shape
    idx = np.arange(W - 1, T)
    Xseq = np.stack([X[i - W + 1:i + 1] for i in idx], axis=0)
    yseq = None if y is None else y[idx]
    return Xseq.astype(np.float32), (None if y is None else yseq.astype(np.float32))

def r2(yhat, y):
    num = ((yhat - y) ** 2).sum()
    den = (y ** 2).sum()
    return 1.0 - float(num / den)
