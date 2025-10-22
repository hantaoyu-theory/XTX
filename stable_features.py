import numpy as np

# ===== Basic helpers =====

def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

# ===== Per-tick features (levels 1..use_levels) =====

def build_tick_features(askRate, bidRate, askSize, bidSize, askNc, bidNc, use_levels: int = 4):
    """
    Builds enhanced LOB features with original + new orthogonal features.
    Inputs are arrays of shape (T, L). We clip to the first `use_levels`.
    Returns X_now with shape (T, F) where F depends on which features are enabled.
    """
    L = use_levels
    A = np.nan_to_num(askRate[:, :L], copy=False)
    B = np.nan_to_num(bidRate[:, :L], copy=False)
    Sa = np.nan_to_num(askSize[:, :L], copy=False)
    Sb = np.nan_to_num(bidSize[:, :L], copy=False)
    Na = np.nan_to_num(askNc[:, :L], copy=False)
    Nb = np.nan_to_num(bidNc[:, :L], copy=False)

    # ============================================
    # ORIGINAL FEATURES (Your current working 6)
    # ============================================
    
    # Queue imbalance for level 0 (best bid/ask)
    qi_level0 = _safe_div(Sb[:, 0] - Sa[:, 0], Sb[:, 0] + Sa[:, 0] + 1e-8)
    
    # Queue imbalance for level 1 (second best bid/ask)
    if L > 1:
        qi_level1 = _safe_div(Sb[:, 1] - Sa[:, 1], Sb[:, 1] + Sa[:, 1] + 1e-8)
    else:
        qi_level1 = np.zeros(Sb.shape[0])
    
    # Top-of-book pressure (bid0/ask0)
    top_book_pressure = Sb[:, 0] / (Sa[:, 0] + 1e-8)
    
    # Total depth
    dep_a = Sa.sum(1)
    dep_b = Sb.sum(1)
    
    # Total volume imbalance
    volume_imb = _safe_div(dep_b - dep_a, dep_a + dep_b)
    
    # Ask book slope
    def slope(P, S):
        cs = np.cumsum(S, axis=1)
        x = cs - cs.mean(axis=1, keepdims=True)
        y = P - P.mean(axis=1, keepdims=True)
        num = (x * y).sum(1)
        den = (x * x).sum(1)
        return _safe_div(num, den)
    
    slope_a = slope(A, Sa)
    
    # Multi-level imbalance (mean over top use_levels)
    multi_level_imb = np.mean((Sb - Sa) / (Sb + Sa + 1e-8), axis=1)
    
    # Depth ratio (sum bid / sum ask)
    depth_ratio = dep_b / (dep_a + 1e-8)
    
    # ============================================
    # NEW ORTHOGONAL FEATURES
    # ============================================
    
    # 1. Liquidity Resilience - depth beyond best quote
    if L > 1:
        level_depth_ratio = Sb[:, 1:].sum(1) / (Sb[:, 0] + 1e-8)
    else:
        level_depth_ratio = np.zeros(Sb.shape[0])
    
    # 2. Price Level Concentration (Herfindahl index)
    bid_sizes_norm = Sb / (Sb.sum(1, keepdims=True) + 1e-8)
    bid_concentration = (bid_sizes_norm ** 2).sum(1)
    
    ask_sizes_norm = Sa / (Sa.sum(1, keepdims=True) + 1e-8)
    ask_concentration = (ask_sizes_norm ** 2).sum(1)
    
    concentration_diff = bid_concentration - ask_concentration
    
    # 3. Asymmetric Depth across levels
    depth_asym_l0 = (Sb[:, 0] - Sa[:, 0]) / (Sb[:, 0] + Sa[:, 0] + 1e-8)
    if L > 1:
        depth_asym_l1 = (Sb[:, 1] - Sa[:, 1]) / (Sb[:, 1] + Sa[:, 1] + 1e-8)
        depth_asym_diff = depth_asym_l0 - depth_asym_l1
    else:
        depth_asym_diff = np.zeros(Sb.shape[0])
    
    # 4. Volume-Weighted Average Price (VWAP) mid
    bid_vwap = (B[:, :L] * Sb[:, :L]).sum(1) / (Sb[:, :L].sum(1) + 1e-8)
    ask_vwap = (A[:, :L] * Sa[:, :L]).sum(1) / (Sa[:, :L].sum(1) + 1e-8)
    vwap_mid = 0.5 * (bid_vwap + ask_vwap)
    
    # 5. Order Size Variance (NC-based)
    avg_size_bid = Sb / (Nb + 1e-8)
    avg_size_ask = Sa / (Na + 1e-8)
    
    # Mean order size ratio
    avg_order_size_bid = avg_size_bid.mean(1)
    avg_order_size_ask = avg_size_ask.mean(1)
    avg_size_ratio = avg_order_size_bid / (avg_order_size_ask + 1e-8)
    
    # Coefficient of variation in order sizes
    bid_size_cv = avg_size_bid.std(1) / (avg_size_bid.mean(1) + 1e-8)
    ask_size_cv = avg_size_ask.std(1) / (avg_size_ask.mean(1) + 1e-8)
    size_cv_ratio = bid_size_cv / (ask_size_cv + 1e-8)
    
    # 6. Relative Spread to Depth
    spread = A[:, 0] - B[:, 0]
    total_depth = Sb[:, 0] + Sa[:, 0]
    spread_to_depth = spread / (total_depth + 1e-8)
    
    # 7. Mid-to-Micro Deviation
    simple_mid = 0.5 * (A[:, 0] + B[:, 0])
    microprice = _safe_div(A[:, 0] * Sb[:, 0] + B[:, 0] * Sa[:, 0], Sa[:, 0] + Sb[:, 0])
    mid_micro_dev = (microprice - simple_mid) / (simple_mid + 1e-8)
    
    # 8. Level Spacing (book tightness)
    if L > 1:
        bid_spacing = (B[:, 0] - B[:, 1]) / (B[:, 0] + 1e-8)
        ask_spacing = (A[:, 1] - A[:, 0]) / (A[:, 0] + 1e-8)
        spacing_ratio = bid_spacing / (ask_spacing + 1e-8)
    else:
        spacing_ratio = np.ones(B.shape[0])
    
    # ============================================
    # OTHER ORIGINAL FEATURES (commented by default)
    # ============================================
    
    # Level 1 pressure
    if L > 1:
        top_book_pressure_l1 = Sb[:, 1] / (Sa[:, 1] + 1e-8)
    else:
        top_book_pressure_l1 = np.zeros(Sb.shape[0])
    
    # Bid book slope
    slope_b = slope(B, Sb)
    
    # Order Flow Imbalance proxy
    ofi_proxy = np.zeros_like(Sa[:, 0])
    if Sa.shape[0] > 1:
        ofi_proxy[1:] = (Sb[1:, 0] - Sb[:-1, 0]) - (Sa[1:, 0] - Sa[:-1, 0])
    
    # Relative spread
    mid_prices = 0.5 * (A[:, 0] + B[:, 0])
    relative_spread = _safe_div(spread, mid_prices)
    
    # Change in imbalance (momentum)
    qi_change = np.zeros_like(qi_level0)
    if len(qi_level0) > 1:
        qi_change[1:] = qi_level0[1:] - qi_level0[:-1]
    
    # Average order size ratio (original NC feature)
    avg_order_bid_orig = np.sum(Sb, axis=1) / (np.sum(Nb, axis=1) + 1e-8)
    avg_order_ask_orig = np.sum(Sa, axis=1) / (np.sum(Na, axis=1) + 1e-8)
    order_size_ratio_orig = avg_order_bid_orig / (avg_order_ask_orig + 1e-8)
    
    # ============================================
    # COMBINE FEATURES
    # ============================================
    
    X_now = np.column_stack([
        # ===== YOUR CURRENT WORKING 6 FEATURES =====
        qi_level0,              # 0: Queue imbalance at level 0
        qi_level1,              # 1: Queue imbalance at level 1
        top_book_pressure,      # 2: Top-of-book pressure
        volume_imb,             # 3: Total volume imbalance
        slope_a,                # 4: Ask book slope
        multi_level_imb,        # 5: Mean imbalance across levels
        depth_ratio,            # 6: Total bid/ask ratio
        
        # ===== NEW ORTHOGONAL FEATURES (8 features) =====
        level_depth_ratio,      # 7: Liquidity resilience (use)
        concentration_diff,     # 8: Concentration difference (use)
        # depth_asym_diff         # 9: Depth asymmetry difference (don't use)
        # vwap_mid                # 10: VWAP mid price (don't use)
        # avg_size_ratio        # 11: Mean order size ratio (NC) (don't use)
        size_cv_ratio,         # 12: Order size variance ratio (NC) (use)
        # spread_to_depth        # 13: Spread to depth ratio (don't use)
        mid_micro_dev,          # 14: Mid-micro deviation (use)
        spacing_ratio,          # 15: Level spacing ratio (use)
        
        # ===== UNCOMMENT TO ADD MORE FEATURES =====
        # top_book_pressure_l1   # Level 1 pressure (don't use)
        # slope_b                # Bid book slope (don't use)
        # ofi_proxy              # Order Flow Imbalance (don't use)
        # relative_spread,        # Relative spread
        # qi_change              # Imbalance momentum
        order_size_ratio_orig  # Original NC ratio
    ])
    
    return X_now  # (T, 16) with current selection

