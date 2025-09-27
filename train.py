# train.py
import argparse, os, numpy as np, torch
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
    ap.add_argument('--window', type=int, default=120)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    args = ap.parse_args()

    # Load arrays from CSV/CSV.GZ/NPZ
    askR, bidR, askS, bidS, askN, bidN, y = load_any(args.data, L_expected=8, has_y=True)

    # Features (strictly causal)
    X_now = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=args.use_levels)
    X = add_causal_temporal_features(X_now, windows=(5, 20, 60))

    # Chronological split
    T = X.shape[0]
    split = int(T * (1.0 - args.val_frac))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # Scale on train only
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # Windows
    W = args.window
    Xtr, ytr = make_sequences(X_tr, y_tr, W)
    Xva, yva = make_sequences(X_val, y_val, W)

    # (N, W, F) -> (N, F, W)
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

    best = (-1e9, None)
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val R^2
        model.eval()
        yh, yt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                yh.append(pred); yt.append(yb.numpy())
        yh = np.concatenate(yh); yt = np.concatenate(yt)
        r2_va = r2(yh, yt)
        print(f"epoch {ep}  val R2={r2_va:.4f}")
        if r2_va > best[0]:
            best = (r2_va, {k: v.cpu() for k, v in model.state_dict().items()})

    os.makedirs(args.outdir, exist_ok=True)
    torch.save(best[1], os.path.join(args.outdir, 'model.pt'))
    dump(scaler, os.path.join(args.outdir, 'scaler.joblib'))
    with open(os.path.join(args.outdir, 'meta.txt'), 'w') as f:
        f.write(f"use_levels={args.use_levels}\nwindow={args.window}\n")

if __name__ == '__main__':
    main()
