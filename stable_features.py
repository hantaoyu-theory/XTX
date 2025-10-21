import numpy as np

# ===== Basic helpers =====

def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

# ===== Per-tick features (levels 1..use_levels) =====

def build_tick_features(askRate, bidRate, askSize, bidSize, askNc, bidNc, use_levels: int = 4):
    """
    Builds 14 carefully selected LOB features.
    Inputs are arrays of shape (T, L). We clip to the first `use_levels`.
    Returns X_now with shape (T, 14).
    """
    L = use_levels
    A = np.nan_to_num(askRate[:, :L], copy=False)
    B = np.nan_to_num(bidRate[:, :L], copy=False)
    Sa = np.nan_to_num(askSize[:, :L], copy=False)
    Sb = np.nan_to_num(bidSize[:, :L], copy=False)
    Na = np.nan_to_num(askNc[:, :L], copy=False)
    Nb = np.nan_to_num(bidNc[:, :L], copy=False)

    # ===== Core Imbalance Features =====
    
    # Queue imbalance for level 0 (best bid/ask)
    qi_level0 = _safe_div(Sb[:, 0] - Sa[:, 0], Sb[:, 0] + Sa[:, 0] + 1e-8)
    
    # Queue imbalance for level 1 (second best bid/ask)
    if Sb.shape[1] > 1 and Sa.shape[1] > 1:
        qi_level1 = _safe_div(Sb[:, 1] - Sa[:, 1], Sb[:, 1] + Sa[:, 1] + 1e-8)
    else:
        qi_level1 = np.zeros(Sb.shape[0])
    
    # ===== Pressure Ratios =====
    
    # Top-of-book pressure (bid0/ask0)
    top_book_pressure = Sb[:, 0] / (Sa[:, 0] + 1e-8)
    
    # Level 1 pressure
    if Sb.shape[1] > 1 and Sa.shape[1] > 1:
        top_book_pressure_l1 = Sb[:, 1] / (Sa[:, 1] + 1e-8)
    else:
        top_book_pressure_l1 = np.zeros(Sb.shape[0])
    
    # ===== Aggregate Volume Features =====
    
    # Total depth
    dep_a = Sa.sum(1)
    dep_b = Sb.sum(1)
    
    # Total volume imbalance
    volume_imb = _safe_div(dep_b - dep_a, dep_a + dep_b)
    
    # ===== Book Shape Features =====
    
    # Book slopes
    def slope(P, S):
        cs = np.cumsum(S, axis=1)
        x = cs - cs.mean(axis=1, keepdims=True)
        y = P - P.mean(axis=1, keepdims=True)
        num = (x * y).sum(1)
        den = (x * x).sum(1)
        return _safe_div(num, den)
    
    slope_a = slope(A, Sa)
    slope_b = slope(B, Sb)
    
    # ===== Order Flow =====
    
    # Order Flow Imbalance proxy
    ofi_proxy = np.zeros_like(Sa[:, 0])
    if Sa.shape[0] > 1:
        ofi_proxy[1:] = (Sb[1:, 0] - Sb[:-1, 0]) - (Sa[1:, 0] - Sa[:-1, 0])
    
    # ===== Price Features =====
    
    # Best prices
    best_a = A[:, 0]
    best_b = B[:, 0]
    
    # Microprice (size-weighted mid)
    top_sz_a = Sa[:, 0]
    top_sz_b = Sb[:, 0]
    microprice = _safe_div(best_a * top_sz_b + best_b * top_sz_a, top_sz_a + top_sz_b)
    
    # ===== Multi-level Aggregates =====
    
    # Multi-level imbalance (mean over top use_levels)
    multi_level_imb = np.mean((Sb - Sa) / (Sb + Sa + 1e-8), axis=1)
    
    # Depth ratio (sum bid / sum ask)
    depth_ratio = dep_b / (dep_a + 1e-8)
    
    # ===== Spread Features =====
    
    # Spread and relative spread
    spread = best_a - best_b
    mid_prices = 0.5 * (best_a + best_b)
    relative_spread = _safe_div(spread, mid_prices)
    
    # ===== Temporal Features =====
    
    # Change in imbalance (momentum)
    qi_change = np.zeros_like(qi_level0)
    if len(qi_level0) > 1:
        qi_change[1:] = qi_level0[1:] - qi_level0[:-1]
    
    # ===== NC-based Features =====
    
    # Average order size ratio
    avg_order_bid = np.sum(Sb, axis=1) / (np.sum(Nb, axis=1) + 1e-8)
    avg_order_ask = np.sum(Sa, axis=1) / (np.sum(Na, axis=1) + 1e-8)
    order_size_ratio = avg_order_bid / (avg_order_ask + 1e-8)
    
    # ===== Combine Features =====
    
    X_now = np.column_stack([
        # Core imbalances (2)
        qi_level0,              # 0: Queue imbalance at level 0
        qi_level1,              # 1: Queue imbalance at level 1
        
        # Pressure ratios (2)
        top_book_pressure,      # 2: Top-of-book pressure
        # top_book_pressure_l1,   # 3: Level 1 pressure
        
        # Volume features (1)
        volume_imb,             # 4: Total volume imbalance
        
        # Book shape (2)
        slope_a,                # 5: Ask book slope
        # slope_b,                # 6: Bid book slope
        
        # Order flow (1)
        # ofi_proxy,              # 7: Order Flow Imbalance proxy
        
        # Price features (1)
        # microprice,             # 8: Size-weighted mid price
        
        # # Multi-level (2)
        multi_level_imb,        # 9: Mean imbalance across levels
        depth_ratio            # 10: Total bid/ask ratio
        
        # # Spread (1)
        # relative_spread,        # 11: Normalized spread
        
        # # Temporal (1)
        # qi_change,              # 12: Change in imbalance
        
        # # NC-based (1)
        # order_size_ratio        # 13: Average order size ratio
    ])
    
    return X_now  # (T, 14)
