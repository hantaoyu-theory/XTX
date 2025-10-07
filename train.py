"""Minimal cumulative training (expanding prefixes) with strict causality.

K=1 behaves like standard split; K>1 reuses one model across K phases.
"""
import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump

from utils import load_any, make_sequences, r2
from features import build_tick_features
from model import LOBTransformer


def _time_weights(kind: str, pos: torch.Tensor, total: int) -> torch.Tensor:
    if kind == 'linear':
        w = 0.5 + 0.5 * (pos.float() / max(total, 1))
    elif kind == 'exponential':
        w = torch.exp(2.0 * pos.float() / max(total, 1))
    else:
        w = torch.ones_like(pos, dtype=torch.float32)
    return w / w.mean()


def train_phase(model, opt, scheduler, device, X_tr, y_tr, X_val, y_val, args, tag: str, scaler: StandardScaler | None):
    # Fit/Update scaler
    if args.scaler_mode == 'cumulative':
        if scaler is None:
            scaler = StandardScaler().fit(X_tr)
        else:
            scaler.partial_fit(X_tr)
    else:
        scaler = StandardScaler().fit(X_tr)

    Xtr, ytr = make_sequences(scaler.transform(X_tr), y_tr, args.window)
    Xva, yva = make_sequences(scaler.transform(X_val), y_val, args.window)

    Xtr_t = torch.from_numpy(Xtr).float().transpose(1, 2).contiguous()
    Xva_t = torch.from_numpy(Xva).float().transpose(1, 2).contiguous()
    ytr_t = torch.from_numpy(ytr).float()
    yva_t = torch.from_numpy(yva).float()

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    N_tr = len(ytr_t)
    loss_fn = torch.nn.MSELoss(reduction='none')

    for ep in range(1, args.epochs + 1):
        model.train()
        for step, (xb, yb) in enumerate(tr_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if args.time_weighting == 'none':
                loss = loss_fn(pred, yb).mean()
            else:
                losses = loss_fn(pred, yb)
                pos = torch.arange(step * args.batch, step * args.batch + len(pred), device=device)
                tw = _time_weights(args.time_weighting, pos, N_tr)
                loss = (losses * tw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if scheduler is not None:
            scheduler.step()

        # Minimal eval on val
        model.eval()
        yh, yt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yh.append(model(xb).cpu().numpy()); yt.append(yb.numpy())
        yh = np.concatenate(yh) if yh else np.empty((0,), dtype=np.float32)
        yt = np.concatenate(yt) if yt else np.empty((0,), dtype=np.float32)
        val_r2 = r2(yh, yt) if len(yt) else float('nan')
        # Log LR if available
        lr = opt.param_groups[0]['lr']
        print(f"[Epoch {ep} {tag}] Val R²={val_r2:.5f} | LR={lr:.2e}")

    return scaler, float(val_r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--use_levels', type=int, default=4)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--purge', action='store_true', default=True)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=2)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    ap.add_argument('--time_weighting', type=str, default='linear', choices=['none','linear','exponential'])
    # Stability/phase controls
    ap.add_argument('--restart_per_phase', type=str, default='scheduler', choices=['none','scheduler','optimizer','both'], help='Whether to restart scheduler/optimizer at each phase boundary')
    ap.add_argument('--sched', type=str, default='cosine', choices=['cosine','warm_restarts'], help='LR scheduler type')
    ap.add_argument('--scaler_mode', type=str, default='per_phase', choices=['per_phase','cumulative'], help='Fit StandardScaler per phase or update cumulatively')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    K = args.k if args.k else 1

    # Load + features
    askR, bidR, askS, bidS, askN, bidN, y = load_any(args.data, L_expected=8, has_y=True)
    X = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=args.use_levels)
    T = X.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = LOBTransformer(in_feats=X.shape[1], d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dim_ff=args.ff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Global scheduler if not restarting each phase
    if args.restart_per_phase in ('none',):
        if args.sched == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * K)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.epochs, T_mult=K)
    else:
        scheduler = None

    purge_amt = (args.window - 1) if args.purge else 0
    metrics = {}
    last_scaler = None

    for phase in range(1, K + 1):
        prefix_end = int(T * (phase / K))
        split_prefix = int(prefix_end * (1.0 - args.val_frac))
        train_end_raw = split_prefix
        train_end = max(0, train_end_raw - purge_amt)
        X_tr = X[:train_end]; y_tr = y[:train_end]
        X_val = X[train_end_raw:prefix_end]; y_val = y[train_end_raw:prefix_end]
        print(f"[PHASE {phase}/{K}] prefix [0:{prefix_end}) train [0:{train_end}) val [{train_end_raw}:{prefix_end})")

        # Optionally restart optimizer and/or scheduler per phase
        if phase > 1:
            if args.restart_per_phase in ('optimizer','both'):
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            if args.restart_per_phase in ('scheduler','both'):
                if args.sched == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.epochs)

        last_scaler, val_r2 = train_phase(model, opt, scheduler, device, X_tr, y_tr, X_val, y_val, args, tag=f"CV{phase}", scaler=last_scaler if args.scaler_mode=='cumulative' else None)
        metrics[f"cv{phase}"] = val_r2

    torch.save({'model': model.state_dict(), 'config': vars(args)}, os.path.join(args.outdir, 'final_model.pt'))
    if last_scaler is not None:
        dump(last_scaler, os.path.join(args.outdir, 'scaler.pkl'))
    with open(os.path.join(args.outdir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[FINAL] Saved to {os.path.join(args.outdir, 'final_model.pt')}")


if __name__ == '__main__':
    main()
