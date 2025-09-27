# predict.py
import argparse, os, numpy as np, torch
from joblib import load
from utils import load_any
from features import build_tick_features, add_causal_temporal_features
from model import LOBTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)        # .csv/.csv.gz/.npz WITHOUT y
    ap.add_argument('--artifacts', required=True)   # dir with model.pt, scaler.joblib, meta.txt
    ap.add_argument('--out', required=True)         # output .npy path
    ap.add_argument('--use_levels', type=int, default=4)  # fallback if meta.txt missing
    args = ap.parse_args()

    # Load meta (window/use_levels) if present
    meta = {}
    meta_path = os.path.join(args.artifacts, 'meta.txt')
    if os.path.exists(meta_path):
        for line in open(meta_path):
            k, v = line.strip().split('=')
            meta[k] = int(v)
    use_levels = meta.get('use_levels', args.use_levels)
    W = meta.get('window', 120)

    # Load arrays (y will be None on test)
    askR, bidR, askS, bidS, askN, bidN, _ = load_any(args.data, L_expected=8, has_y=False)

    # Build features (strictly causal) and scale
    X_now = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=use_levels)
    X = add_causal_temporal_features(X_now, windows=(5, 20, 60))
    scaler = load(os.path.join(args.artifacts, 'scaler.joblib'))
    X = scaler.transform(X)

    # Window into sequences
    T, F = X.shape
    idx = np.arange(W - 1, T)
    if len(idx) == 0:
        raise ValueError(f"Not enough timesteps ({T}) for window size {W}.")
    Xseq = np.stack([X[i - W + 1:i + 1] for i in idx], axis=0).astype(np.float32)  # (N, W, F)
    Xseq = np.transpose(Xseq, (0, 2, 1))  # (N, F, W)

    # Load model
    state = torch.load(os.path.join(args.artifacts, 'model.pt'), map_location='cpu')
    model = LOBTransformer(in_feats=Xseq.shape[1])
    model.load_state_dict(state)
    model.eval()

    # Predict
    yhat_chunks = []
    with torch.no_grad():
        B = 1024
        for i in range(0, Xseq.shape[0], B):
            xb = torch.from_numpy(Xseq[i:i + B])
            pred = model(xb).numpy()
            yhat_chunks.append(pred)
    yhat = np.concatenate(yhat_chunks)

    # Align to original timeline (first W-1 are NaN)
    out = np.full((T,), np.nan, dtype=np.float32)
    out[idx] = yhat
    np.save(args.out, out)

if __name__ == '__main__':
    main()
