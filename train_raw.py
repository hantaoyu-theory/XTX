#!/usr/bin/env python3

import argparse
from sklearn.preprocessing import StandardScaler
import yaml
from utils import load_any, make_sequences, r2
from model import LOBTransformer
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd


def apply_time_weighting(losses, time_weighting='none', alpha=0.1):
    """Apply time-based weighting to losses."""
    if time_weighting == 'none':
        return losses
    
    n = len(losses)
    if time_weighting == 'linear':
        weights = torch.linspace(1.0, 1.0 + alpha, n, device=losses.device)
    elif time_weighting == 'exponential':
        weights = torch.exp(alpha * torch.arange(n, device=losses.device, dtype=torch.float32) / n)
    else:
        raise ValueError(f"Unknown time weighting: {time_weighting}")
    
    return losses * weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--use_levels', type=int, default=4)  # From train.py (not used but for compatibility)
    parser.add_argument('--window', type=int, default=10)     # From train.py (not used but for compatibility)
    parser.add_argument('--batch', type=int, default=512)     # Changed from batch_size to batch (like train.py)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--ff', type=int, default=256)
    parser.add_argument('--time_weighting', type=str, default='exponential', 
                       choices=['none', 'linear', 'exponential'])
    parser.add_argument('--sample_frac', type=float, default=None, help='Sample fraction of data')
    
    args = parser.parse_args()
    
    # Load raw data 
    print("Loading raw data...")
    if args.data.endswith('.gz'):
        df = pd.read_csv(args.data, compression='gzip')
    else:
        df = pd.read_csv(args.data)
    
    print(f"Original data shape: {df.shape}")
    
    # Sample if requested
    if args.sample_frac and args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).sort_index()
        print(f"Sampled data shape: {df.shape}")
    
    # Use only top 2 levels (12 features total)
    top2_features = []
    for level in range(2):  # 0, 1 (top 2 levels)
        top2_features.extend([
            f'askRate_{level}', f'bidRate_{level}',
            f'askSize_{level}', f'bidSize_{level}', 
            f'askNc_{level}', f'bidNc_{level}'
        ])
    
    # Select only these 12 features
    X = df[top2_features].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    
    print(f"Selected top-2 features: {len(top2_features)} columns")
    print(f"Feature columns: {top2_features}")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Create sequences
    print("Creating sequences...")
    X_seq, y_seq = make_sequences(X, y, args.window)
    print(f"Sequences shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Split data
    split_idx = int((1 - args.val_frac) * len(X_seq))
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Convert to PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(y_val).to(device)
    
    # Create model - input size is number of selected features (12)
    input_size = len(top2_features)
    model = LOBTransformer(
        input_size=input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ff_dim=args.ff,
        dropout=args.dropout,
        seq_len=args.window
    ).to(device)
    
    print(f"Model input size: {input_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss(reduction='none')
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['input_size'] = input_size
    config['top2_features'] = top2_features
    with open(os.path.join(args.outdir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Training loop
    best_r2 = float('-inf')
    
    for epoch in range(args.epochs):
        model.train()
        
        # Training
        train_losses = []
        for i in range(0, len(X_train), args.batch):
            batch_X = X_train[i:i+args.batch]
            batch_y = y_train[i:i+args.batch]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate per-sample losses
            losses = criterion(outputs, batch_y)
            
            # Apply time weighting if specified
            if args.time_weighting != 'none':
                losses = apply_time_weighting(losses, args.time_weighting)
            
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training R²
            train_outputs = model(X_train)
            train_r2 = r2(y_train.cpu().numpy(), train_outputs.cpu().numpy())
            
            # Validation R²
            val_outputs = model(X_val)
            val_r2 = r2(y_val.cpu().numpy(), val_outputs.cpu().numpy())
        
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={avg_train_loss:.6f}, Train R²={train_r2:.6f}, Val R²={val_r2:.6f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'config': config,
                'r2': val_r2,
                'epoch': epoch
            }, os.path.join(args.outdir, 'best_model.pt'))
    
    print(f"Training completed. Best R²: {best_r2:.6f}")


if __name__ == '__main__':
    main()
