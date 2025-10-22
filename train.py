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


def get_feature_names_from_stable_features():
    """
    Parse stable_features.py to extract feature names from the code.
    Returns list of feature names in the order they appear in np.column_stack.
    """
    try:
        with open('stable_features.py', 'r') as f:
            content = f.read()
        
        # Find the np.column_stack section
        if 'np.column_stack([' in content:
            start = content.find('np.column_stack([')
            end = content.find('])', start)
            stack_section = content[start:end]
            
            # Extract variable names (lines with commas that aren't comments)
            feature_names = []
            for line in stack_section.split('\n'):
                # Remove inline comments first
                if '#' in line:
                    line = line.split('#')[0]
                
                line = line.strip()
                
                # Skip empty lines, the column_stack line itself, and comment-only lines
                if not line or 'np.column_stack' in line:
                    continue
                
                # Remove trailing comma
                var_name = line.rstrip(',').strip()
                
                # Only add if it's a valid variable name (not empty, not a bracket)
                if var_name and var_name not in ['[', ']', '']:
                    feature_names.append(var_name)
            
            return feature_names
    except Exception as e:
        print(f"Warning: Could not parse feature names from stable_features.py: {e}")
        return None

def get_feature_abbreviations(num_features):
    """
    Generate feature abbreviations based on feature names from stable_features.py.
    Falls back to generic names if parsing fails.
    """
    feature_names = get_feature_names_from_stable_features()
    
    if feature_names is None or len(feature_names) != num_features:
        # Fallback to generic names
        print(f"Using generic feature names (could not parse or mismatch)")
        feature_names = [f'feat{i}' for i in range(num_features)]
    
    # Create abbreviations: remove underscores, take first 3 chars, uppercase
    feature_abbr = [str(name).replace('_', '')[:3].upper() for name in feature_names]
    return feature_abbr, feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='train.csv.gz')
    ap.add_argument('--outdir', default='results_14feat')
    ap.add_argument('--use_levels', type=int, default=4)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=2)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--ff', type=int, default=256)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--wandb_project', default='lob-transformer')
    ap.add_argument('--wandb_name', default=None)
    ap.add_argument('--train_start', type=float, default=0.0)
    ap.add_argument('--train_end', type=float, default=0.48)
    ap.add_argument('--val_end', type=float, default=0.6)
    ap.add_argument('--ratio', type=float, default=0.3)  # Time weighting ratio
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    askR, bidR, askS, bidS, askN, bidN, y = load_any(args.data, L_expected=8, has_y=True)
    print(f"Loaded {len(y)} timesteps with {askR.shape[1]} levels")

    print("Building enhanced features with NC data...")
    X = build_tick_features(askR, bidR, askS, bidS, askN, bidN, use_levels=args.use_levels)
    T = X.shape[0]
    num_of_features = X.shape[1]
    print(f"Features shape: {X.shape} (T={T}, F={num_of_features})")

    # Automatically detect feature names by parsing stable_features.py
    feature_abbr, feature_names = get_feature_abbreviations(num_of_features)
    feature_str = '_'.join(feature_abbr)
    print(f"Detected {num_of_features} features: {feature_names}")
    print(f"Feature abbreviations: {feature_str}")
    
    # Auto-generate experiment name with feature abbreviations
    if args.wandb_name is None:
        args.wandb_name = f"feat_lr{args.lr:.0e}_w{args.window}_l{args.layers}_r{args.ratio:.1f}_start{args.train_start:.2f}_end{args.train_end:.2f}_val{args.val_end:.2f}_levels{args.use_levels}_dmodel{args.d_model}_f{feature_str}"
    
    print(f"Experiment name: {args.wandb_name}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )
    
    # Calculate split indices
    train_start_idx = int(T * args.train_start)
    train_end_idx = int(T * args.train_end)
    val_end_idx = int(T * args.val_end)
    
    # Train/val split based on provided fractions
    X_train = X[train_start_idx:train_end_idx]
    y_train = y[train_start_idx:train_end_idx]
    X_val = X[train_end_idx:val_end_idx]
    y_val = y[train_end_idx:val_end_idx]
    
    print(f"Data split:")
    print(f"  Train: [{train_start_idx:,} : {train_end_idx:,}] = {len(X_train):,} samples ({args.train_start:.1%} - {args.train_end:.1%})")
    print(f"  Val:   [{train_end_idx:,} : {val_end_idx:,}] = {len(X_val):,} samples ({args.train_end:.1%} - {args.val_end:.1%})")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler().fit(X_train)
    
    # Create sequences
    print(f"Creating sequences with window={args.window}...")
    Xtr, ytr = make_sequences(scaler.transform(X_train), y_train, args.window)
    Xva, yva = make_sequences(scaler.transform(X_val), y_val, args.window)

    print(f"Train sequences shape: {Xtr.shape} (B={len(ytr)}, T={args.window}, F={num_of_features})")
    print(f"Val sequences shape: {Xva.shape} (B={len(yva)}, T={args.window}, F={num_of_features})")
    
    # Create time-based weights for training data
    # Weight increases linearly from (1-ratio) to 1.0 across training sequences
    time_indices_tr = np.arange(len(ytr))
    min_t, max_t = time_indices_tr.min(), time_indices_tr.max()
    time_weights_tr = (time_indices_tr - min_t) / (max_t - min_t + 1e-8) * args.ratio + (1 - args.ratio)
    
    print(f"Time weights: min={time_weights_tr.min():.3f}, max={time_weights_tr.max():.3f}, ratio={args.ratio}")
    
    # Convert to tensors
    Xtr_t = torch.from_numpy(Xtr).float()  
    Xva_t = torch.from_numpy(Xva).float()
    ytr_t = torch.from_numpy(ytr).float()
    yva_t = torch.from_numpy(yva).float()
    weights_t = torch.from_numpy(time_weights_tr).float()  # (N_tr,) - sample weights
    
    print(f"Tensor shapes - X: {Xtr_t.shape}, weights: {weights_t.shape}, y: {ytr_t.shape}")
    
    # Create data loaders - include weights for training
    train_dataset = TensorDataset(Xtr_t, ytr_t, weights_t)
    val_dataset = TensorDataset(Xva_t, yva_t)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)  # Don't shuffle to preserve time order
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LOBTransformer(
        in_feats=num_of_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.ff
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training variables
    best_val_r2 = -float('inf')
    patience_counter = 0
    best_model_state = None
    train_r2_history = []
    val_r2_history = []
    epoch_list = []
    
    # Add batch-level tracking for plotting
    batch_val_r2_history = []  # Store validation R² every 100 batches
    batch_numbers = []  # Store global batch numbers for x-axis
    global_batch_counter = 0  # Track total batches across all epochs
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        total_train_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            xb, yb, wb = [b.to(device) for b in batch]  # No time tensor
            global_batch_counter += 1  # Increment global batch counter

            opt.zero_grad()
            pred = model(xb)  # No time argument

            # Calculate per-sample losses
            losses = torch.nn.functional.mse_loss(pred, yb, reduction='none')

            # Apply time weights (later data more important)
            weighted_losses = losses * wb
            loss = weighted_losses.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()
            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(yb.cpu().numpy())

            # Progress bar
            progress = (batch_idx + 1) / total_train_batches
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)
            print(f'\r[Train Epoch {epoch:2d}] [{bar}] {progress*100:5.1f}% | Batch {batch_idx+1}/{total_train_batches} | Loss: {loss.item():.6f}', 
                  end='', flush=True)

            # Every 100 batches, evaluate on first 20 validation batches
            if global_batch_counter % 100 == 0:
                model.eval()
                val_preds_sample = []
                val_targets_sample = []
                
                with torch.no_grad():
                    # Just iterate through first 20 batches - much faster!
                    for i, batch_val in enumerate(val_loader):
                        if i >= 20:  # Only take first 20 batches
                            break
                        xb_val, yb_val = [b.to(device) for b in batch_val]
                        pred_val = model(xb_val)  # No time argument
                        val_preds_sample.append(pred_val.cpu().numpy())
                        val_targets_sample.append(yb_val.cpu().numpy())
                
                if val_preds_sample and val_targets_sample:
                    val_preds_sample = np.concatenate(val_preds_sample)
                    val_targets_sample = np.concatenate(val_targets_sample)
                    val_r2_sample = r2(val_preds_sample, val_targets_sample)
                    
                    # Store for plotting
                    batch_val_r2_history.append(val_r2_sample)
                    batch_numbers.append(global_batch_counter)
                    
                    # Log to wandb
                    wandb.log({
                        'val_r2_batch': val_r2_sample,
                        'batch_number': global_batch_counter
                    })
                
                model.train()  # Back to training mode
        
        print()  # New line after training
        
        # Calculate training metrics
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_r2 = r2(train_preds, train_targets)
        train_r2_history.append(train_r2)
        avg_train_loss = train_loss / total_train_batches
        
        # Full validation at end of epoch
        model.eval()
        val_preds = []
        val_targets = []
        
        total_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                xb, yb = [b.to(device) for b in batch]  # No time tensor
                
                pred = model(xb)  # No time argument
                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
                
                # Progress bar
                progress = (batch_idx + 1) / total_val_batches
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '=' * filled + '-' * (bar_length - filled)
                print(f'\r[Val Epoch {epoch:2d}]   [{bar}] {progress*100:5.1f}% | Batch {batch_idx+1}/{total_val_batches}', 
                      end='', flush=True)
        
        print()  # New line after validation
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_r2 = r2(val_preds, val_targets)
        val_mse = np.mean((val_preds - val_targets) ** 2)
        val_r2_history.append(val_r2)
        epoch_list.append(epoch)
        
        print(f"\n[Epoch {epoch:2d} Summary]")
        print(f"  Train Loss: {avg_train_loss:.6f} | Train R²: {train_r2:.5f}")
        print(f"  Val R²: {val_r2:.5f} | Val MSE: {val_mse:.6f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'val_mse': val_mse,
            'best_val_r2': best_val_r2,
            'patience_counter': patience_counter
        })
        
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
            print(f"\nEarly stopping triggered! Best Val R²: {best_val_r2:.5f}")
            break
    
    # Create two plots: epoch-level and batch-level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Epoch-level R² (original)
    ax1.plot(epoch_list, train_r2_history, label='Train R²', marker='o')
    ax1.plot(epoch_list, val_r2_history, label='Val R²', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² per Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Batch-level validation R² (every 100 batches)
    if batch_val_r2_history:  # Only plot if we have data
        ax2.plot(batch_numbers, batch_val_r2_history, label='Val R² (first 20 batches)', 
                 color='green', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Validation R² Every 100 Batches')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for epoch boundaries
        batches_per_epoch = len(train_loader)
        for ep in range(1, epoch + 1):
            ax2.axvline(x=ep * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'r2_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"R² curves saved to {plot_path}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val R²: {best_val_r2:.5f}")
    
    # Save model and artifacts
    torch.save({
        'model': model.state_dict(),
        'config': vars(args),
        'n_features': num_of_features,
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_end': args.val_end,
        'ratio': args.ratio,
        'feature_names': feature_names
    }, os.path.join(args.outdir, 'final_model.pt'))
    
    dump(scaler, os.path.join(args.outdir, 'scaler.pkl'))
    
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({
            'final_val_r2': float(best_val_r2),
            'final_val_mse': float(val_mse),
            'n_features': num_of_features,
            'feature_names': feature_names,
            'train_start': args.train_start,
            'train_end': args.train_end,
            'val_end': args.val_end,
            'ratio': args.ratio
        }, f, indent=2)
    
    # Log final results to wandb
    wandb.log({
        'final_val_r2': float(best_val_r2),
        'final_val_mse': float(val_mse)
    })
    
    if os.path.exists(plot_path):
        wandb.log({"r2_curves": wandb.Image(plot_path)})
    
    wandb.finish()
    
    print(f"\n[FINAL] Val R²: {best_val_r2:.5f}, saved to {args.outdir}")


if __name__ == '__main__':
    main()