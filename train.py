"""Training script for LOB temporal transformer with forward-chaining CV and time-weighted loss."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump

from utils import load_any, make_sequences, r2
from features import build_tick_features
from model import LOBTransformer


# Simple helper to train on one split (mirrors original k=1 flow)
def train_split(args, X_tr_np, y_tr_np, X_val_np, y_val_np, outdir_path, split_tag=""):
    os.makedirs(outdir_path, exist_ok=True)

    # Fit scaler on train only
    scaler_local = StandardScaler()
    X_tr_scaled = scaler_local.fit_transform(X_tr_np)
    X_val_scaled = scaler_local.transform(X_val_np)

    # Windowed sequences
    W_local = args.window
    Xtr_np, ytr_np = make_sequences(X_tr_scaled, y_tr_np, W_local)
    Xva_np, yva_np = make_sequences(X_val_scaled, y_val_np, W_local)

    # Tensors (N, F, W) expected by model
    Xtr_t = torch.from_numpy(Xtr_np).float().transpose(1, 2).contiguous()
    Xva_t = torch.from_numpy(Xva_np).float().transpose(1, 2).contiguous()
    ytr_t = torch.from_numpy(ytr_np).float()
    yva_t = torch.from_numpy(yva_np).float()

    # Loaders
    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(Xva_t, yva_t), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} {split_tag}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = LOBTransformer(
        in_feats=Xtr_t.shape[1], d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dim_ff=args.ff
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss(reduction="none")

    N_tr = len(ytr_t)
    for ep in range(1, args.epochs + 1):
        model.train()
        total_steps = len(train_loader)
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if args.time_weighting == "none":
                loss = loss_fn(pred, yb).mean()
            else:
                losses = loss_fn(pred, yb)
                batch_start_idx = step * args.batch
                sample_positions = torch.arange(
                    batch_start_idx, batch_start_idx + len(pred), device=device
                )
                total_samples = N_tr
                if args.time_weighting == "linear":
                    time_weights = 0.5 + 0.5 * (sample_positions.float() / max(total_samples, 1))
                elif args.time_weighting == "exponential":
                    time_weights = torch.exp(2.0 * sample_positions.float() / max(total_samples, 1))
                else:
                    time_weights = torch.ones_like(sample_positions, dtype=torch.float32)
                time_weights = time_weights / time_weights.mean()
                loss = (losses * time_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            print(
                f"[Train] Epoch {ep}/{args.epochs} progress: {100.0 * (step + 1) / total_steps:.2f}% ({step + 1}/{total_steps})",
                end="\r",
            )

        # Eval
        model.eval()
        yh_train, yt_train = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yh_train.append(model(xb).cpu().numpy())
                yt_train.append(yb.numpy())
        yh_train = np.concatenate(yh_train)
        yt_train = np.concatenate(yt_train)
        train_r2 = r2(yh_train, yt_train)

        # Train MSEs and tail R2
        try:
            pos = np.arange(N_tr, dtype=np.float32)
            if args.time_weighting == "linear":
                tw = 0.5 + 0.5 * (pos / max(N_tr, 1))
            elif args.time_weighting == "exponential":
                tw = np.exp(2.0 * (pos / max(N_tr, 1)))
            else:
                tw = np.ones(N_tr, dtype=np.float32)
            tw = tw / (tw.mean() if tw.mean() != 0 else 1.0)
            se_train = (yh_train - yt_train) ** 2
            train_mse_unweighted = float(se_train.mean())
            train_mse_weighted = float(np.average(se_train, weights=tw))
            tail_n = max(1, int(0.2 * N_tr))
            train_tail_r2 = r2(yh_train[-tail_n:], yt_train[-tail_n:])
        except Exception:
            train_mse_unweighted = float("nan")
            train_mse_weighted = float("nan")
            train_tail_r2 = float("nan")

        yh_val, yt_val = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yh_val.append(model(xb).cpu().numpy())
                yt_val.append(yb.numpy())
        yh_val = np.concatenate(yh_val)
        yt_val = np.concatenate(yt_val)
        r2_epoch = r2(yh_val, yt_val)
        val_mse = float(((yh_val - yt_val) ** 2).mean())

        print(
            f"\n[Epoch {ep}{(' ' + split_tag) if split_tag else ''}] Train R²={train_r2:.5f}, Val R²={r2_epoch:.5f}"
        )
        print(
            f"[Epoch {ep}{(' ' + split_tag) if split_tag else ''}] Train MSE(w)={train_mse_weighted:.6f}, "
            f"Train MSE={train_mse_unweighted:.6f}, Val MSE={val_mse:.6f}, TrainTail R²={train_tail_r2:.5f}"
        )
        scheduler.step()

    print(
        f"\n[Final{(' ' + split_tag) if split_tag else ''}] Training completed. Final validation R²={r2_epoch:.5f}"
    )

    # Save model weights and scaler
    torch.save({"model": model.state_dict(), "config": vars(args)}, os.path.join(outdir_path, "final_model.pt"))
    dump(scaler_local, os.path.join(outdir_path, "scaler.pkl"))
    print(f"[INFO] Model saved to {outdir_path}/final_model.pt")
    try:
        with open(os.path.join(outdir_path, "metrics.json"), "w") as f:
            json.dump({"val_r2": float(r2_epoch), "config": vars(args)}, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json to {outdir_path}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)  # .csv/.csv.gz/.npz
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use_levels", type=int, default=4)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument(
        "--time_weighting",
        type=str,
        default="linear",
        choices=["none", "linear", "exponential"],
        help="How to weight samples by time",
    )
    # Time-series cross-validation controls
    ap.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of forward-chaining CV folds (1 = no CV, use full data split).",
    )
    ap.add_argument(
        "--purge",
        action="store_true",
        default=True,
        help="In CV mode, drop W-1 samples before the validation block to avoid window overlap leakage.",
    )
    args = ap.parse_args()

    # Ensure output directory exists for sweep/runs
    os.makedirs(args.outdir, exist_ok=True)

    # Normalize folds count
    K = args.k if (args.k is not None) else 1

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
    print(f"X shape after features: {X.shape}")
    print("[INFO] Features built. Splitting and scaling data...")

    # Time-series split logic: standard (K=1) or run K phases when K>1
    T = X.shape[0]
    if K <= 1:
        split = int(T * (1.0 - args.val_frac))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        print(
            f"[SPLIT] Standard split: Train [0:{split}) ({split} ticks), Val [{split}:{T}) ({T - split} ticks)"
        )
        train_split(args, X_tr, y_tr, X_val, y_val, args.outdir, split_tag="STD")
    else:
        purge_amt = (args.window - 1) if args.purge else 0
        for fold in range(1, K + 1):
            fold_end = int(T * (fold / K))
            if fold_end <= 0:
                raise ValueError("Fold end computed as 0; check k setting")
            split_prefix = int(fold_end * (1.0 - args.val_frac))
            train_end_raw = split_prefix
            train_end = max(0, train_end_raw - purge_amt)
            X_tr_f, y_tr_f = X[:train_end], y[:train_end]
            X_val_f, y_val_f = X[train_end_raw:fold_end], y[train_end_raw:fold_end]
            print(
                f"[CV] Fold {fold}/{K}: Prefix [0:{fold_end}) | Train [0:{train_end}) (purged {purge_amt}) | Val [{train_end_raw}:{fold_end})"
            )
            outdir_fold = os.path.join(args.outdir, f"cv{fold}")
            train_split(args, X_tr_f, y_tr_f, X_val_f, y_val_f, outdir_fold, split_tag=f"CV{fold}")

    # In CV mode, we've already run and saved each fold inside the loop.


if __name__ == "__main__":
    main()
