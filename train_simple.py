"""Simple training with 80/20 split using stable features - no phases."""
import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import wandb

from utils import load_any, make_sequences, r2
from stable_features import build_tick_features
from model import LOBTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='train.csv.gz')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--use_levels', type=int, default=4)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=2)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    ap.add_argument('--ratio', type=float, default=0.7)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--wandb_project', default='lob-transformer')
    ap.add_argument('--wandb_name', default=None)
    ap.add_argument('--data_frac', type=float, default=0.6)
    ap.add_argument('--training_frac', type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data and extract features
    askR, bidR, askS, bidS, askN, bidN, y = load_any(args.data, L_expected=8, has_y=True)
    X = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=args.use_levels)

    # Flexible feature abbreviation: use first 3 letters of each feature name if available
    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
    else:
        feature_names = [f'f{i}' for i in range(X.shape[1])]
    feature_abbr = [str(name)[:3].upper() for name in feature_names]
    feature_str = '_'.join(feature_abbr)
    # Auto-generate experiment name from parameters if not provided
    if args.wandb_name is None:
        args.wandb_name = f"stable_w{args.window}_b{args.batch}_lr{args.lr:.0e}_d{args.d_model}_h{args.nhead}_l{args.layers}_ff{args.ff}_r{args.ratio}_df{args.data_frac}_tf{args.training_frac}_f{feature_str}"

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )
    
    data_subset_idx = int(args.data_frac * X.shape[0])
    X = X[:data_subset_idx]
    y = y[:data_subset_idx]
    T = X.shape[0]


    split_idx = int(args.training_frac * T)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"Train samples: {len(X_tr)}, Val samples: {len(X_val)}")


    scaler = StandardScaler().fit(X_tr)
    
    Xtr, ytr = make_sequences(scaler.transform(X_tr), y_tr, args.window)
    Xva, yva = make_sequences(scaler.transform(X_val), y_val, args.window)
    
    time_indices_tr = np.arange(args.window - 1, len(X_tr))[:len(ytr)]
    min_t, max_t = time_indices_tr.min(), time_indices_tr.max()
    time_weights_tr = (time_indices_tr - min_t) / (max_t - min_t) * args.ratio + (1-args.ratio)

    # Convert to tensors
    Xtr_t = torch.from_numpy(Xtr).float()
    Xva_t = torch.from_numpy(Xva).float()
    ytr_t = torch.from_numpy(ytr).float()
    yva_t = torch.from_numpy(yva).float()
    weights_t = torch.from_numpy(time_weights_tr).float()

    # Data loaders (include time weights for training)
    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t, weights_t), batch_size=args.batch, shuffle=False)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=args.batch, shuffle=False)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LOBTransformer(in_feats=X.shape[1], d_model=args.d_model, nhead=args.nhead, 
                          num_layers=args.layers, dim_ff=args.ff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Training on {device}")

    # Track metrics for plotting (per batch)
    train_r2_history = []
    val_r2_history = []
    batch_numbers = []
    global_batch_count = 0
    
    # Early stopping variables
    best_val_r2 = -float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for ep in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        total_steps = len(tr_loader)
        for step, (xb, yb, wb) in enumerate(tr_loader):
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            pred = model(xb)
            
            # Calculate per-sample losses
            losses = torch.nn.functional.mse_loss(pred, yb, reduction='none')
            
            # Apply time weights (later data more important)
            weighted_losses = losses * wb.unsqueeze(1)  # unsqueeze to match loss dimensions
            loss = weighted_losses.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            
            # Collect predictions for R²
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(yb.cpu().numpy())
            
            global_batch_count += 1
            
            # Evaluate after every 100 batches or at end of epoch
            if (step + 1) % 100 == 0 or step == total_steps - 1:
                # Calculate current training R²
                current_train_preds = np.concatenate(train_preds)
                current_train_targets = np.concatenate(train_targets)
                current_train_r2 = r2(current_train_preds, current_train_targets)
                
                # Quick validation evaluation
                model.eval()
                val_preds_batch, val_targets_batch = [], []
                with torch.no_grad():
                    for i, (xb_val, yb_val) in enumerate(va_loader):
                        if i >= 10:  # Only use first 10 batches for speed
                            break
                        xb_val = xb_val.to(device)
                        val_preds_batch.append(model(xb_val).cpu().numpy())
                        val_targets_batch.append(yb_val.numpy())
                
                val_preds_sample = np.concatenate(val_preds_batch)
                val_targets_sample = np.concatenate(val_targets_batch)
                current_val_r2 = r2(val_preds_sample, val_targets_sample)
                
                # Store for plotting
                batch_numbers.append(global_batch_count)
                train_r2_history.append(current_train_r2)
                val_r2_history.append(current_val_r2)
                
                # Log to wandb
                wandb.log({
                    'batch': global_batch_count,
                    'epoch': ep,
                    'train_r2_batch': current_train_r2,
                    'val_r2_batch': current_val_r2
                })
                
                model.train()  # Back to training mode
            
            # Live progress
            pct = (step + 1) * 100.0 / total_steps
            print(f"\r[Train] Epoch {ep}/{args.epochs} progress: {pct:6.2f}% ({step+1}/{total_steps}) | Batch {global_batch_count}", end='', flush=True)
        print()  # newline after epoch
        
        # Calculate training R²
        train_preds_all = np.concatenate(train_preds)
        train_targets_all = np.concatenate(train_targets)
        train_r2 = r2(train_preds_all, train_targets_all)

        # Evaluate
        model.eval()
        yh, yt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yh.append(model(xb).cpu().numpy())
                yt.append(yb.numpy())

        yh = np.concatenate(yh)
        yt = np.concatenate(yt)
        val_r2 = r2(yh, yt)
        val_mse = np.mean((yh - yt) ** 2)
        train_loss_avg = train_loss / len(tr_loader)

        print(f"[Epoch {ep:2d}] Train Loss: {train_loss_avg:.6f}, Train R²: {train_r2:.5f}, Val R²: {val_r2:.5f}, Val MSE: {val_mse:.6f}")

        # Log to wandb
        wandb.log({
            'epoch': ep,
            'train_loss': train_loss_avg,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'val_mse': val_mse,
            'best_val_r2': best_val_r2,
            'patience_counter': patience_counter
        })

        # Note: batch-level metrics are now stored during training loop
        
        # Early stopping check
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"    -> New best Val R²: {best_val_r2:.5f}")
        else:
            patience_counter += 1
            print(f"    -> No improvement for {patience_counter}/{args.patience} epochs")
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered! Best Val R²: {best_val_r2:.5f}")
            break

    # Create and save R² plot (per batch)
    plt.figure(figsize=(10, 6))
    plt.plot(batch_numbers, train_r2_history, 'b-', label='Train R²', linewidth=2)
    plt.plot(batch_numbers, val_r2_history, 'r-', label='Validation R²', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('R² Score')
    plt.title('Training and Validation R² Over Batches (with Time Weighting)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.outdir, 'r2_curve_weighted.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"R² curve saved to {plot_path}")

    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val R²: {best_val_r2:.5f}")
    
    # Save model
    torch.save({'model': model.state_dict(), 'config': vars(args)}, 
               os.path.join(args.outdir, 'final_model.pt'))
    dump(scaler, os.path.join(args.outdir, 'scaler.pkl'))
    
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({'final_val_r2': float(best_val_r2), 'final_val_mse': float(val_mse)}, f, indent=2)
    
    # Log final metrics to wandb
    wandb.log({
        'final_val_r2': float(best_val_r2),
        'final_val_mse': float(val_mse),
        'total_batches': global_batch_count,
        'total_epochs': ep
    })
    
    # Upload plot to wandb
    if os.path.exists(plot_path):
        wandb.log({"r2_curve": wandb.Image(plot_path)})
    
    wandb.finish()
    
    print(f"[FINAL] Val R²: {best_val_r2:.5f}, saved to {args.outdir}")


if __name__ == '__main__':
    main()
