# train.py
import argparse, os, json, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump
from utils import load_any, make_sequences, r2
from features import build_tick_features, add_causal_temporal_features
from model import LOBTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)              # .csv/.csv.gz/.npz
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--use_levels', type=int, default=4)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    args = ap.parse_args()

    # Ensure output directory exists for sweep/runs
    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Starting data extraction...")
    askR, bidR, askS, bidS, askN, bidN, y = load_any(args.data, L_expected=8, has_y=True)
    print(f"askRate shape: {askR.shape}")
    print(f"bidRate shape: {bidR.shape}")
    print(f"askSize shape: {askS.shape}")
    print(f"bidSize shape: {bidS.shape}")
    print(f"askNc shape: {askN.shape}")
    print(f"bidNc shape: {bidN.shape}")
    print(f"y shape: {y.shape}")
    print("[INFO] Data loaded. Building features...")

    X = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=args.use_levels)
    
    print(f"X shape after temporal features: {X.shape}")
    print("[INFO] Features built. Splitting and scaling data...")


    T = X.shape[0]
    n_folds = 4  # You can change this
    fold_size = T // (n_folds + 1)
    val_scores = []
    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = val_start + fold_size
        if val_end > T:
            break
        print(f"[CV] Fold {fold+1}/{n_folds}: train 0:{train_end}, val {val_start}:{val_end}")
        X_tr, X_val = X[:train_end], X[val_start:val_end]
        y_tr, y_val = y[:train_end], y[val_start:val_end]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        W = args.window
        Xtr, ytr = make_sequences(X_tr, y_tr, W)
        Xva, yva = make_sequences(X_val, y_val, W)

        Xtr = torch.from_numpy(np.transpose(Xtr, (0, 2, 1)))
        Xva = torch.from_numpy(np.transpose(Xva, (0, 2, 1)))
        ytr = torch.from_numpy(ytr)
        yva = torch.from_numpy(yva)

        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=False)
        val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LOBTransformer(in_feats=Xtr.shape[1], d_model=args.d_model, nhead=args.nhead,
                               num_layers=args.layers, dim_ff=args.ff).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.MSELoss()

        for ep in range(1, args.epochs + 1):
            model.train()
            total_steps = len(train_loader)
            for step, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                progress = 100.0 * (step + 1) / total_steps
                print(f"[CV Fold {fold+1}] Epoch {ep}/{args.epochs} Training progress: {progress:.2f}% ({step + 1}/{total_steps})", end='\r')

        # Validation for this fold
        model.eval()
        yh, yt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                yh.append(pred); yt.append(yb.numpy())
        yh = np.concatenate(yh); yt = np.concatenate(yt)
        r2_va = r2(yh, yt)
        print(f"[CV Fold {fold+1}] val R2={r2_va:.4f}")
        val_scores.append(r2_va)

    avg_r2 = float(np.mean(val_scores))
    print(f"[CV] Average val R2 over {n_folds} folds: {avg_r2:.4f}")

    # Persist simple metrics + config for sweep consumption
    try:
        metrics = {
            "fold_r2": [float(x) for x in val_scores],
            "avg_r2": avg_r2,
            "config": vars(args),
        }
        with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json to {args.outdir}: {e}")

if __name__ == '__main__':
    main()
