# train.py
import argparse, os, json, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
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
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    ap.add_argument('--time_weighting', type=str, default='exponential', 
                    choices=['none', 'linear', 'exponential'], 
                    help='How to weight samples by time')
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
    print(f"X shape after features: {X.shape}")
    print("[INFO] Features built. Splitting and scaling data...")


    # Simple 80/20 split (controlled by --val_frac)
    T = X.shape[0]
    split = int(T * (1.0 - args.val_frac))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    W = args.window
    Xtr, ytr = make_sequences(X_tr, y_tr, W)
    Xva, yva = make_sequences(X_val, y_val, W)

    # Optimize tensor creation: contiguous, proper dtype, efficient transpose
    Xtr = torch.from_numpy(Xtr).float().transpose(1, 2).contiguous()  # (N, F, W)
    Xva = torch.from_numpy(Xva).float().transpose(1, 2).contiguous()  # (N, F, W) 
    ytr = torch.from_numpy(ytr).float()
    yva = torch.from_numpy(yva).float()

    # Single-process DataLoaders to avoid memory issues from worker multiplication
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, 
                             shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch, 
                           shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable cudnn auto-tuning for fixed input shapes (5-15% speedup)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    model = LOBTransformer(in_feats=Xtr.shape[1], d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.layers, dim_ff=args.ff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    # Time-weighted loss: later samples get higher weight
    loss_fn = torch.nn.MSELoss(reduction='none')  # Per-sample losses

    for ep in range(1, args.epochs + 1):
        model.train()
        total_steps = len(train_loader)
        for step, (xb, yb) in enumerate(train_loader):
            # Non-blocking transfer (overlaps with compute)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)  # Faster than zero_grad()
            pred = model(xb)
            
            # Time-weighted loss based on user choice
            if args.time_weighting == 'none':
                loss = loss_fn(pred, yb).mean()
            else:
                losses = loss_fn(pred, yb)  # Per-sample losses (batch_size,)
                
                # Calculate global position of each sample in this batch
                batch_start_idx = step * args.batch
                sample_positions = torch.arange(batch_start_idx, batch_start_idx + len(pred), device=device)
                total_samples = len(ytr)  # Total training samples
                
                # Create weights based on time position
                if args.time_weighting == 'linear':
                    # Linear: weight = 0.5 + 0.5 * (position / total)
                    time_weights = 0.5 + 0.5 * (sample_positions.float() / total_samples)
                elif args.time_weighting == 'exponential':  
                    # Exponential: later samples get much higher weight
                    time_weights = torch.exp(2.0 * sample_positions.float() / total_samples)
                
                # Normalize weights to maintain loss scale
                time_weights = time_weights / time_weights.mean()
                loss = (losses * time_weights).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            progress = 100.0 * (step + 1) / total_steps
            print(f"[Train] Epoch {ep}/{args.epochs} progress: {progress:.2f}% ({step + 1}/{total_steps})", end='\r')
        
        # Evaluate on both training and validation sets after each epoch
        model.eval()
        
        # Training R²
        yh_train, yt_train = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                pred = model(xb).cpu().numpy()
                yh_train.append(pred); yt_train.append(yb.numpy())
        yh_train = np.concatenate(yh_train); yt_train = np.concatenate(yt_train)
        train_r2 = r2(yh_train, yt_train)
        
        # Validation R²
        yh_val, yt_val = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                pred = model(xb).cpu().numpy()
                yh_val.append(pred); yt_val.append(yb.numpy())
        yh_val = np.concatenate(yh_val); yt_val = np.concatenate(yt_val)
        r2_epoch = r2(yh_val, yt_val)
        
        print(f"\n[Epoch {ep}] Train R²={train_r2:.5f}, Val R²={r2_epoch:.5f}")
        
        scheduler.step()  # Update learning rate

    # Final validation (already computed in last epoch, but for consistency)
    print(f"\n[Final] Training completed. Final validation R²={r2_epoch:.5f}")

    # Save trained model and scaler (REQUIRED by task)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'model_class': 'LOBTransformer',
        'scaler_state': scaler,
        'val_r2': r2_epoch
    }, os.path.join(args.outdir, 'final_model.pt'))
    
    print(f"[INFO] Model saved to {args.outdir}/final_model.pt")

    # Persist simple metrics + config for sweep consumption
    avg_r2 = float(r2_epoch)
    try:
        metrics = {
            "val_r2": avg_r2,
            "config": vars(args),
        }
        with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json to {args.outdir}: {e}")

if __name__ == '__main__':
    main()
