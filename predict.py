# predict.py
import argparse, os, json, numpy as np, torch
from joblib import load
from utils import load_any
from features import build_tick_features
from model import LOBTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)        # .csv/.csv.gz/.npz WITHOUT y
    # Preferred: load a single training checkpoint (final_model.pt or best_model.pt)
    ap.add_argument('--checkpoint', required=False, help='Path to .pt saved by training (contains model_state_dict, config, scaler_state)')
    # Fallback legacy mode: artifacts directory with model.pt + scaler.joblib + meta.txt
    ap.add_argument('--artifacts', required=False, help='Directory with model.pt, scaler.joblib, meta.txt')
    ap.add_argument('--out', required=True)         # output .npy path (aligned to T with leading NaNs)
    # Optional overrides when missing in checkpoint
    ap.add_argument('--use_levels', type=int, default=None)
    ap.add_argument('--window', type=int, default=None)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Determine mode: checkpoint (.pt with scaler+config) or legacy artifacts dir
    ckpt = None
    cfg = {}
    scaler = None
    use_levels = None
    W = None
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        cfg = ckpt.get('config', {})
        scaler = ckpt.get('scaler_state', None)
        if scaler is None:
            raise RuntimeError('Checkpoint missing scaler_state; cannot standardize features.')
        use_levels = args.use_levels if args.use_levels is not None else int(cfg.get('use_levels', 4))
        W = args.window if args.window is not None else int(cfg.get('window', 10))
    elif args.artifacts and os.path.isdir(args.artifacts):
        # Legacy mode
        meta = {}
        meta_path = os.path.join(args.artifacts, 'meta.txt')
        if os.path.exists(meta_path):
            for line in open(meta_path):
                k, v = line.strip().split('=')
                meta[k] = int(v)
        use_levels = args.use_levels if args.use_levels is not None else meta.get('use_levels', 4)
        W = args.window if args.window is not None else meta.get('window', 10)
        scaler = load(os.path.join(args.artifacts, 'scaler.joblib'))
    else:
        raise ValueError('Provide either --checkpoint path or --artifacts directory')

    # Load arrays (y will be None on test)
    askR, bidR, askS, bidS, askN, bidN, _ = load_any(args.data, L_expected=8, has_y=False)

    # Build features (strictly causal) and scale
    # Build the same features as training (no extra temporal features here)
    X = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=use_levels)
    X = scaler.transform(X)

    # Window into sequences
    T, F = X.shape
    idx = np.arange(W - 1, T)
    if len(idx) == 0:
        raise ValueError(f"Not enough timesteps ({T}) for window size {W}.")
    # Sliding windows; to keep memory modest we’ll iterate in batches during inference
    # We still need aligned output with length T (NaNs for first W-1)

    # Load model
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Instantiate model
    in_feats = X.shape[1]
    if ckpt is not None:
        d_model = int(cfg.get('d_model', 128))
        nhead = int(cfg.get('nhead', 4))
        layers = int(cfg.get('layers', 2))
        dim_ff = int(cfg.get('ff', 256))
        model = LOBTransformer(in_feats=in_feats, d_model=d_model, nhead=nhead,
                               num_layers=layers, dim_ff=dim_ff).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        state = torch.load(os.path.join(args.artifacts, 'model.pt'), map_location='cpu')
        model = LOBTransformer(in_feats=in_feats).to(device)
        model.load_state_dict(state)
    model.eval()

    # Predict
    # Batched sliding-window inference without materializing all windows
    yhat_parts = []
    B = int(args.batch)
    with torch.no_grad():
        for start in range(W - 1, T, B):
            end = min(T, start + B)
            # Build a block of windows [start .. end-1]
            block = np.stack([X[i - W + 1:i + 1] for i in range(start, end)], axis=0).astype(np.float32)  # (b, W, F)
            xb = torch.from_numpy(np.transpose(block, (0, 2, 1))).to(device, non_blocking=True)  # (b, F, W)
            pred = model(xb).cpu().numpy()
            yhat_parts.append(pred)
    yhat = np.concatenate(yhat_parts, axis=0)

    # Align to original timeline (first W-1 are NaN)
    out = np.full((T,), np.nan, dtype=np.float32)
    out[idx] = yhat
    # Save predictions
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if args.out.endswith('.npy'):
        np.save(args.out, out)
    else:
        # Default to .npy
        np.save(args.out, out)
        
    # Also store a small manifest for traceability
    manifest = {
        'data': os.path.abspath(args.data),
        'out': os.path.abspath(args.out),
        'window': int(W),
        'use_levels': int(use_levels),
        'mode': 'checkpoint' if ckpt is not None else 'artifacts',
        'config': cfg,
    }
    try:
        with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write manifest: {e}")

if __name__ == '__main__':
    main()
