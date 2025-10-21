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

    # Book pressure (imbalance) for levels 0, 1, 2, 3
    lvl0_imb = _safe_div(Sb[:, 0] - Sa[:, 0], Sb[:, 0] + Sa[:, 0] + 1e-8)
    if Sb.shape[1] > 1 and Sa.shape[1] > 1:
        lvl1_imb = _safe_div(Sb[:, 1] - Sa[:, 1], Sb[:, 1] + Sa[:, 1] + 1e-8)
    else:
        lvl1_imb = np.zeros(Sb.shape[0])
    if Sb.shape[1] > 2 and Sa.shape[1] > 2:
        lvl2_imb = _safe_div(Sb[:, 2] - Sa[:, 2], Sb[:, 2] + Sa[:, 2] + 1e-8)
    else:
        lvl2_imb = np.zeros(Sb.shape[0])
    if Sb.shape[1] > 3 and Sa.shape[1] > 3:
        lvl3_imb = _safe_div(Sb[:, 3] - Sa[:, 3], Sb[:, 3] + Sa[:, 3] + 1e-8)
    else:
        lvl3_imb = np.zeros(Sb.shape[0])

    # Level 1 features
    if Sb.shape[1] > 1 and Sa.shape[1] > 1:
        top_book_pressure_l1 = Sb[:, 1] / (Sa[:, 1] + 1e-8)
        spread_l1 = A[:, 1] - B[:, 1]
    else:
        top_book_pressure_l1 = np.zeros(Sb.shape[0])
        spread_l1 = np.zeros(Sb.shape[0])
    # Level 2 features
    if Sb.shape[1] > 2 and Sa.shape[1] > 2:
        qi_level2 = _safe_div(Sb[:, 2] - Sa[:, 2], Sb[:, 2] + Sa[:, 2] + 1e-8)
        top_book_pressure_l2 = Sb[:, 2] / (Sa[:, 2] + 1e-8)
        spread_l2 = A[:, 2] - B[:, 2]
    else:
        qi_level2 = np.zeros(Sb.shape[0])
        top_book_pressure_l2 = np.zeros(Sb.shape[0])
        spread_l2 = np.zeros(Sb.shape[0])

    # Level 3 features
    if Sb.shape[1] > 3 and Sa.shape[1] > 3:
        qi_level3 = _safe_div(Sb[:, 3] - Sa[:, 3], Sb[:, 3] + Sa[:, 3] + 1e-8)
        top_book_pressure_l3 = Sb[:, 3] / (Sa[:, 3] + 1e-8)
        spread_l3 = A[:, 3] - B[:, 3]
    else:
        qi_level3 = np.zeros(Sb.shape[0])
        top_book_pressure_l3 = np.zeros(Sb.shape[0])
        spread_l3 = np.zeros(Sb.shape[0])

    # Order Flow Imbalance (OFI) proxy: change in total bid/ask size
    ofi_proxy = np.zeros_like(Sa[:,0])
    if Sa.shape[0] > 1:
        ofi_proxy[1:] = (Sb[1:,0] - Sb[:-1,0]) - (Sa[1:,0] - Sa[:-1,0])

    # Short-term price volatility (rolling std of mid price, window=10)
    mid_prices = 0.5 * (A[:,0] + B[:,0])
    try:
        import pandas as pd
        price_vol10 = pd.Series(mid_prices).rolling(window=10, min_periods=1).std().values
    except ImportError:
        price_vol10 = np.zeros_like(mid_prices)

    best_a = A[:, 0]                          # best ask price
    best_b = B[:, 0]                          # best bid price
    spread = best_a - best_b                  # spread
    # Relative position of mid price (distance to best bid normalized by spread)
    rel_mid_pos = (mid_prices - best_b) / (spread + 1e-8)

    # Spread volatility (rolling std of spread, window=10)
    try:
        spread_vol10 = pd.Series(spread).rolling(window=10, min_periods=1).std().values
    except ImportError:
        spread_vol10 = np.zeros_like(spread)

    # Multi-level imbalance (mean over top use_levels)
    multi_level_imb = np.mean((Sb - Sa) / (Sb + Sa + 1e-8), axis=1)
    # Depth ratio (sum bid / sum ask)
    cum_bid = Sb.sum(axis=1)
    cum_ask = Sa.sum(axis=1)
    depth_ratio = cum_bid / (cum_ask + 1e-8)
    # Top-of-book pressure (bid0/ask0)
    top_book_pressure = Sb[:,0] / (Sa[:,0] + 1e-8)

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

    # Enhanced feature set for better performance
    volume_imb = _safe_div(dep_b - dep_a, dep_a + dep_b)  # Total volume imbalance
    price_impact = _safe_div(spread, mid_prices)  # Relative spread

    # Queue imbalance for level 1 (second best bid/ask)
    if Sb.shape[1] > 1 and Sa.shape[1] > 1:
        qi_level1 = _safe_div(Sb[:, 1] - Sa[:, 1], Sb[:, 1] + Sa[:, 1] + 1e-8)
    else:
        qi_level1 = np.zeros(Sb.shape[0])
    
    # Level-wise features (use more levels)
    lvl_spreads = A - B  # Spread at each level (T, L)
    lvl_mids = 0.5 * (A + B)  # Mid at each level (T, L)
    
    # Weighted features
    total_ask_vol = Sa.sum(axis=1)
    total_bid_vol = Sb.sum(axis=1)
    weighted_ask_price = _safe_div((A * Sa).sum(axis=1), total_ask_vol)
    weighted_bid_price = _safe_div((B * Sb).sum(axis=1), total_bid_vol)
    
    # More sophisticated imbalance
    top3_ask_vol = Sa[:, :3].sum(axis=1) 
    top3_bid_vol = Sb[:, :3].sum(axis=1)
    top3_imb = _safe_div(top3_bid_vol - top3_ask_vol, top3_bid_vol + top3_ask_vol)

    spread_l2 = A[:, 1] - B[:, 1]
    spread_l1 = A[:, 0] - B[:, 0]
    spread_ratio = spread_l2 / (spread_l1 + 1e-8)

    # 2. Mid-price return
    mid = 0.5 * (A[:, 0] + B[:, 0])
    mid_return = np.diff(mid, prepend=mid[0]) / (mid + 1e-8)

    # 3. Imbalance volatility (rolling std of queue imbalance)
    qi = _safe_div(Sb[:, 0] - Sa[:, 0], Sb[:, 0] + Sa[:, 0] + 1e-8)
    try:
        import pandas as pd
        qi_vol = pd.Series(qi).rolling(window=20, min_periods=1).std().values
    except ImportError:
        # fallback: use zeros if pandas not available
        qi_vol = np.zeros_like(qi)

    # Normalized depth imbalance removed


    # Queue imbalance for level 0 (best bid/ask)
    qi_level0 = _safe_div(Sb[:, 0] - Sa[:, 0], Sb[:, 0] + Sa[:, 0] + 1e-8)

    X_now = np.column_stack([
        qi_level0,         # Queue imbalance at level 0
        qi_level1,         # Queue imbalance at level 1
        qi_level2,         # Queue imbalance at level 2
        qi_level3,         # Queue imbalance at level 3
        top_book_pressure, # Top-of-book pressure (level 0)
        top_book_pressure_l1, # Top-of-book pressure (level 1)
        top_book_pressure_l2, # Top-of-book pressure (level 2)
        top_book_pressure_l3, # Top-of-book pressure (level 3)
        lvl0_imb,          # Book pressure at level 0
        lvl1_imb,          # Book pressure at level 1
        lvl2_imb,          # Book pressure at level 2
        lvl3_imb,          # Book pressure at level 3
        volume_imb,        # Total volume imbalance (normalized)
        slope_a,           # Ask book slope
        slope_b,           # Bid book slope
        ofi_proxy          # Order Flow Imbalance proxy
    ])
    return X_now  # (T, 15)