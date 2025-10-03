#!/usr/bin/env python3
"""
MIRACLE PyTorch Implementation - Simple Demo
==========================================

This script demonstrates the basic usage of the MIRACLE imputation method
with verbose output to show the training process.
"""

import numpy as np
import torch
from miracle_pytorch_port import miracle_impute, MiracleTorch, MiracleConfig


def create_demo_data():
    """Create a simple dataset with some structure and missing values."""
    np.random.seed(42)
    
    # Create structured data
    n_samples, n_features = 150, 5
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add some relationships
    X[:, 1] = 0.6 * X[:, 0] + 0.4 * X[:, 1] + np.random.normal(0, 0.2, n_samples)  # Feature 1 depends on feature 0
    X[:, 3] = 0.5 * X[:, 0] + 0.3 * X[:, 2] + 0.2 * X[:, 3]  # Feature 3 depends on features 0 and 2
    
    # Introduce missing values in features 1 and 3
    missing_rate = 0.3
    missing_mask_1 = np.random.rand(n_samples) < missing_rate
    missing_mask_3 = np.random.rand(n_samples) < missing_rate
    
    X_missing = X.copy()
    X_missing[missing_mask_1, 1] = np.nan
    X_missing[missing_mask_3, 3] = np.nan
    
    return X, X_missing, [1, 3]


def demo_basic_usage():
    """Demonstrate basic MIRACLE usage."""
    print("=" * 60)
    print("MIRACLE PyTorch Demo - Basic Usage")
    print("=" * 60)
    
    # Create demo data
    X_true, X_missing, missing_cols = create_demo_data()
    
    print(f"Dataset: {X_missing.shape[0]} samples, {X_missing.shape[1]} features")
    print(f"Missing columns: {missing_cols}")
    print(f"Missing values: {np.isnan(X_missing).sum()} ({np.isnan(X_missing).sum()/(X_missing.size):.1%})")
    
    # Run imputation with verbose output
    print("\n" + "=" * 40)
    print("Running MIRACLE Imputation...")
    print("=" * 40)
    
    X_imputed = miracle_impute(
        X_missing=X_missing,
        missing_list=missing_cols,
        lr=0.01,
        batch_size=32,
        n_hidden=32,
        max_steps=100,
        reg_lambda=1.0,
        reg_beta=1.0,
        reg_m=1.0,
        verbose=True,
        random_seed=42
    )
    
    # Evaluate results
    print("\n" + "=" * 40)
    print("Results")
    print("=" * 40)
    
    for col in missing_cols:
        missing_mask = np.isnan(X_missing[:, col])
        if missing_mask.sum() > 0:
            true_vals = X_true[missing_mask, col]
            imputed_vals = X_imputed[missing_mask, col]
            
            mse = np.mean((true_vals - imputed_vals) ** 2)
            mae = np.mean(np.abs(true_vals - imputed_vals))
            corr = np.corrcoef(true_vals, imputed_vals)[0, 1]
            
            print(f"Column {col}:")
            print(f"  Missing count: {missing_mask.sum()}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Correlation: {corr:.4f}")
            print(f"  True range: [{true_vals.min():.3f}, {true_vals.max():.3f}]")
            print(f"  Imputed range: [{imputed_vals.min():.3f}, {imputed_vals.max():.3f}]")
            print()


def demo_advanced_usage():
    """Demonstrate advanced MIRACLE usage with custom configuration."""
    print("\n" + "=" * 60)
    print("MIRACLE PyTorch Demo - Advanced Usage")
    print("=" * 60)
    
    # Create demo data
    X_true, X_missing, missing_cols = create_demo_data()
    
    # Custom configuration
    config = MiracleConfig(
        lr=0.005,               # Learning rate
        batch_size=16,          # Smaller batches
        num_inputs=X_missing.shape[1],  # Will be updated automatically
        n_hidden=48,            # Larger hidden layer
        reg_lambda=0.5,         # Reduced supervised loss weight
        reg_beta=2.0,           # Increased group lasso
        reg_m=1.5,              # Increased moment regularization
        window=15,              # Larger sliding window
        max_steps=150,          # More training steps
        verbose=True,
        random_seed=123
    )
    
    print("Configuration:")
    for key, value in config.__dict__.items():
        if key != 'verbose':
            print(f"  {key}: {value}")
    
    # Create and train model
    print(f"\nCreating model with {len(missing_cols)} indicator columns...")
    model = MiracleTorch(config, n_indicators=len(missing_cols), missing_list=missing_cols)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Fit the model with early stopping
    print("\nTraining with early stopping...")
    X_imputed = model.fit(
        X_missing=X_missing,
        missing_list=missing_cols,
        early_stopping=True
    )
    
    # Results
    print("\nAdvanced Results:")
    for col in missing_cols:
        missing_mask = np.isnan(X_missing[:, col])
        if missing_mask.sum() > 0:
            true_vals = X_true[missing_mask, col]
            imputed_vals = X_imputed[missing_mask, col]
            
            mse = np.mean((true_vals - imputed_vals) ** 2)
            mae = np.mean(np.abs(true_vals - imputed_vals))
            corr = np.corrcoef(true_vals, imputed_vals)[0, 1]
            
            print(f"  Column {col}: MSE={mse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}")


def main():
    """Run both demos."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Basic demo
        demo_basic_usage()
        
        # Advanced demo
        demo_advanced_usage()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()