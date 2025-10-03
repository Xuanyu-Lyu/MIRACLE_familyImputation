

import numpy as np
import torch
import time
from miracle_pytorch_port import miracle_impute, MiracleTorch, MiracleConfig


def mean_squared_error(y_true, y_pred):
    """Simple MSE implementation."""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Simple MAE implementation.""" 
    return np.mean(np.abs(y_true - y_pred))


def generate_test_data(n_samples=200, n_features=8, missing_rate=0.3, random_seed=42):
    """Generate synthetic test data with controlled missingness."""
    np.random.seed(random_seed)
    
    # Generate base data with some structure - simple multivariate normal
    mean = np.zeros(n_features)
    # Create a correlation matrix with some structure
    cov = np.eye(n_features)
    for i in range(n_features-1):
        cov[i, i+1] = 0.3
        cov[i+1, i] = 0.3
    
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Add some nonlinear relationships
    if n_features > 2:
        X[:, 2] = 0.7 * X[:, 0] + 0.3 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    # Make column 5 partially dependent on columns 1 and 3
    if n_features > 5:
        X[:, 5] = 0.4 * X[:, 1] + 0.4 * X[:, 3] + 0.2 * X[:, 5] + np.random.normal(0, 0.1, n_samples)
    
    return X


def introduce_missing_data(X, missing_columns, missing_rate=0.3, pattern="random"):
    """Introduce missing data with different patterns."""
    X_missing = X.copy()
    n_samples = X.shape[0]
    
    for col in missing_columns:
        if pattern == "random":
            # Random missingness
            mask = np.random.rand(n_samples) < missing_rate
        elif pattern == "systematic":
            # Systematic missingness based on another column
            ref_col = 0 if col != 0 else 1
            threshold = np.percentile(X[:, ref_col], (1 - missing_rate) * 100)
            mask = X[:, ref_col] > threshold
        else:
            mask = np.random.rand(n_samples) < missing_rate
            
        X_missing[mask, col] = np.nan
    
    return X_missing


def evaluate_imputation(X_true, X_imputed, X_missing, missing_columns):
    """Evaluate imputation quality."""
    results = {}
    
    for col in missing_columns:
        # Get indices where values were actually missing
        missing_mask = np.isnan(X_missing[:, col])
        
        if missing_mask.sum() > 0:
            true_vals = X_true[missing_mask, col]
            imputed_vals = X_imputed[missing_mask, col]
            
            mse = mean_squared_error(true_vals, imputed_vals)
            mae = mean_absolute_error(true_vals, imputed_vals)
            
            # Calculate correlation
            correlation = np.corrcoef(true_vals, imputed_vals)[0, 1] if len(true_vals) > 1 else np.nan
            
            results[f'col_{col}'] = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'n_missing': missing_mask.sum()
            }
    
    return results


def test_miracle_basic():
    """Basic functionality test."""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Generate test data
    X_true = generate_test_data(n_samples=100, n_features=6, random_seed=42)
    missing_columns = [1, 3, 4]
    X_missing = introduce_missing_data(X_true, missing_columns, missing_rate=0.25)
    
    print(f"Data shape: {X_true.shape}")
    print(f"Missing columns: {missing_columns}")
    print(f"Total missing values: {np.isnan(X_missing).sum()}")
    print(f"Missing rate: {np.isnan(X_missing).sum() / X_missing.size:.2%}")
    
    # Test with verbose output
    print(f"Data statistics before imputation:")
    print(f"  Min: {np.nanmin(X_missing):.4f}, Max: {np.nanmax(X_missing):.4f}")
    print(f"  Mean: {np.nanmean(X_missing):.4f}, Std: {np.nanstd(X_missing):.4f}")
    
    start_time = time.time()
    X_imputed = miracle_impute(
        X_missing=X_missing,
        missing_list=missing_columns,
        lr=0.001,  # Reduced learning rate for stability
        batch_size=16,
        n_hidden=16,  # Smaller network for stability
        max_steps=50,  # Fewer steps for initial testing
        verbose=True,
        random_seed=42
    )
    end_time = time.time()
    
    print(f"Data statistics after imputation:")
    print(f"  Min: {np.min(X_imputed):.4f}, Max: {np.max(X_imputed):.4f}")
    print(f"  Mean: {np.mean(X_imputed):.4f}, Std: {np.std(X_imputed):.4f}")
    print(f"  Contains NaN: {np.any(np.isnan(X_imputed))}")
    
    print(f"\nImputation completed in {end_time - start_time:.2f} seconds")
    
    # Evaluate results
    results = evaluate_imputation(X_true, X_imputed, X_missing, missing_columns)
    
    print("\nIMPUTATION QUALITY METRICS:")
    for col_name, metrics in results.items():
        print(f"{col_name:>8}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"Corr={metrics['correlation']:.4f}, N_missing={metrics['n_missing']}")
    
    return X_true, X_missing, X_imputed, results


def test_miracle_advanced():
    """Advanced test with different configurations."""
    print("\n" + "=" * 60)
    print("ADVANCED CONFIGURATION TEST")
    print("=" * 60)
    
    X_true = generate_test_data(n_samples=150, n_features=8, random_seed=123)
    missing_columns = [0, 2, 5, 7]
    X_missing = introduce_missing_data(X_true, missing_columns, missing_rate=0.4)
    
    configs = [
        {"name": "Low Regularization", "reg_lambda": 0.1, "reg_beta": 0.1, "reg_m": 0.1},
        {"name": "High Regularization", "reg_lambda": 2.0, "reg_beta": 2.0, "reg_m": 2.0},
        {"name": "DAG Only Mode", "DAG_only": True, "reg_lambda": 1.0},
        {"name": "Large Hidden Layer", "n_hidden": 64, "max_steps": 200},
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        # Extract config name and remove it from parameters
        config_name = config.pop("name")
        
        # Set default parameters
        params = {
            "X_missing": X_missing,
            "missing_list": missing_columns,
            "lr": 0.005,
            "batch_size": 20,
            "n_hidden": 32,
            "max_steps": 150,
            "verbose": False,  # Reduced verbosity for multiple configs
            "random_seed": 456
        }
        
        # Update with specific config
        params.update(config)
        
        start_time = time.time()
        X_imputed = miracle_impute(**params)
        end_time = time.time()
        
        results = evaluate_imputation(X_true, X_imputed, X_missing, missing_columns)
        all_results[config_name] = {
            'results': results,
            'time': end_time - start_time
        }
        
        # Calculate average metrics
        avg_mse = np.mean([r['mse'] for r in results.values()])
        avg_mae = np.mean([r['mae'] for r in results.values()])
        avg_corr = np.mean([r['correlation'] for r in results.values() if not np.isnan(r['correlation'])])
        
        print(f"Time: {end_time - start_time:.2f}s, Avg MSE: {avg_mse:.4f}, "
              f"Avg MAE: {avg_mae:.4f}, Avg Corr: {avg_corr:.4f}")
    
    return all_results


def test_miracle_class_interface():
    """Test the class-based interface with custom configuration."""
    print("\n" + "=" * 60)
    print("CLASS INTERFACE TEST")
    print("=" * 60)
    
    # Generate test data
    X_true = generate_test_data(n_samples=80, n_features=5, random_seed=789)
    missing_columns = [1, 3]
    X_missing = introduce_missing_data(X_true, missing_columns, missing_rate=0.35)
    
    # Create custom configuration
    config = MiracleConfig(
        lr=0.01,
        batch_size=12,
        num_inputs=X_true.shape[1],  # Will be updated by model
        n_hidden=24,
        reg_lambda=1.5,
        reg_beta=0.8,
        reg_m=1.2,
        window=8,
        max_steps=120,
        verbose=True,
        random_seed=321,
        device="cpu"
    )
    
    print(f"Configuration: {config}")
    
    # Create and train model
    n_indicators = len(missing_columns)
    model = MiracleTorch(config, n_indicators=n_indicators, missing_list=missing_columns)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Fit the model
    X_imputed = model.fit(
        X_missing=X_missing,
        missing_list=missing_columns,
        early_stopping=True
    )
    
    # Evaluate
    results = evaluate_imputation(X_true, X_imputed, X_missing, missing_columns)
    
    print("\nFINAL RESULTS:")
    for col_name, metrics in results.items():
        print(f"{col_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
              f"Correlation={metrics['correlation']:.4f}")
    
    return model, results


def test_error_cases():
    """Test error handling and edge cases."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING AND EDGE CASES")
    print("=" * 60)
    
    # Test 1: No missing data
    print("Test 1: No missing data")
    X_complete = np.random.randn(50, 4)
    try:
        X_imputed = miracle_impute(X_complete, missing_list=[], max_steps=10, verbose=True)
        print("✓ Handled complete data correctly")
    except Exception as e:
        print(f"✗ Error with complete data: {e}")
    
    # Test 2: All missing in one column
    print("\nTest 2: Completely missing column")
    X_partial = np.random.randn(50, 4)
    X_partial[:, 2] = np.nan
    try:
        X_imputed = miracle_impute(X_partial, missing_list=[2], max_steps=20, verbose=True)
        print("✓ Handled completely missing column")
        print(f"  Imputed values range: [{X_imputed[:, 2].min():.3f}, {X_imputed[:, 2].max():.3f}]")
    except Exception as e:
        print(f"✗ Error with completely missing column: {e}")
    
    # Test 3: Very small dataset
    print("\nTest 3: Very small dataset")
    X_tiny = np.random.randn(5, 3)
    X_tiny[np.random.rand(5, 3) < 0.4] = np.nan
    missing_cols = [i for i in range(3) if np.isnan(X_tiny[:, i]).any()]
    try:
        X_imputed = miracle_impute(X_tiny, missing_list=missing_cols, max_steps=10, batch_size=2)
        print("✓ Handled tiny dataset")
    except Exception as e:
        print(f"✗ Error with tiny dataset: {e}")


def main():
    """Run comprehensive test suite."""
    print("MIRACLE PyTorch Implementation - Comprehensive Test Suite")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Basic functionality test
        X_true, X_missing, X_imputed, basic_results = test_miracle_basic()
        
        # Advanced configuration tests
        advanced_results = test_miracle_advanced()
        
        # Class interface test
        model, class_results = test_miracle_class_interface()
        
        # Error handling tests
        test_error_cases()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        print("✓ Basic functionality test completed")
        print("✓ Advanced configuration tests completed")
        print("✓ Class interface test completed")
        print("✓ Error handling tests completed")
        
        print(f"\nAdvanced test comparison:")
        for config_name, data in advanced_results.items():
            avg_mse = np.mean([r['mse'] for r in data['results'].values()])
            print(f"  {config_name:>20}: Avg MSE = {avg_mse:.4f}, Time = {data['time']:.2f}s")
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        raise


# ---------------------------- Test execution -------------------------------
if __name__ == "__main__":
    main()
