import os
import yaml
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel as C, WhiteKernel, DotProduct
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_kernel(kernel_type, n_features):
    # Base: Constant (Vertical Scale) + White (Noise/Jitter)
    base = C(1.0, (1e-10, 1e15)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

    if kernel_type == "RBF":
        structure = RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-5, 1e10))
    elif kernel_type == "Matern":
        structure = Matern(length_scale=[1.0] * n_features, nu=2.5, length_scale_bounds=(1e-5, 1e10))
    elif kernel_type == "Linear":
        structure = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e10))
    elif kernel_type == "Quadratic":
        structure = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e10)) ** 2
    else:
        raise ValueError(f"Kernel '{kernel_type}' not supported.")

    return (C(1.0, (1e-10, 1e15)) * structure) + base

def plot_diagnostics(y_true, y_pred, y_std, title, save_dir, exp_id):
    """Generates Parity and Residual plots for the Validation set."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Parity Plot
    plt.figure(figsize=(6, 6))
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    buff = (max_v - min_v) * 0.1
    
    plt.plot([min_v-buff, max_v+buff], [min_v-buff, max_v+buff], 'k--', label='Ideal')
    plt.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='o', alpha=0.5, ecolor='gray', label='Validation Data')
    
    plt.title(f"Parity: {title}")
    plt.xlabel("True Value")
    plt.ylabel("GP Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_id}_parity.png"), dpi=300)
    plt.close()

    # 2. Residual Plot
    residuals = y_true - y_pred
    z_scores = residuals / (y_std + 1e-9)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(z_scores)), z_scores, alpha=0.6)
    plt.axhline(0, c='k', ls='--')
    plt.axhline(2, c='r', ls=':'); plt.axhline(-2, c='r', ls=':')
    plt.title("Residuals vs Index")
    plt.ylabel("Z-score")
    
    plt.subplot(1, 2, 2)
    plt.hist(z_scores, bins=15, density=True, alpha=0.6, color='green')
    plt.title("Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_id}_residuals.png"), dpi=300)
    plt.close()

def plot_gp_slices(gp, X_train, y_train, feature_names, save_dir, exp_id):
    """
    Plots the GP behavior by varying one feature at a time across 1000 points.
    Other features are held at their mean.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_features = X_train.shape[1]
    
    # Calculate the 'Reference Point' (Mean of all inputs)
    mean_inputs = np.mean(X_train, axis=0)
    
    # Setup subplots
    cols = 1
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(n_features):
        ax = axes[i]
        fname = feature_names[i]
        
        # 1. Create Grid (Min to Max of training data)
        x_min, x_max = X_train[:, i].min(), X_train[:, i].max()
        span = x_max - x_min
        # Add small buffer to view edges clearly
        x_grid = np.linspace(x_min - 0.05*span, x_max + 0.05*span, 1000)
        
        # 2. Create Design Matrix (1000 samples, n_features)
        X_design = np.tile(mean_inputs, (1000, 1))
        X_design[:, i] = x_grid
        
        # 3. Predict
        y_pred, y_std = gp.predict(X_design, return_std=True)
        
        # 4. Plot
        
        ax.plot(x_grid, y_pred, 'b-', lw=2, label='GP Mean')
        ax.fill_between(x_grid, y_pred - 1.96*y_std, y_pred + 1.96*y_std, color='b', alpha=0.2, label='95% CI')
        
        # Overlay Training Data (Projected onto this dimension)
        ax.scatter(X_train[:, i], y_train, c='k', s=10, alpha=0.3, label='Training Data')
        
        ax.set_title(exp_id + f" - Varying: {fname}")
        ax.set_xlabel(r"$\Gamma_\gamma(E=4 keV)$, eV")
        ax.set_ylabel("Output")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_id}_slices.png"), dpi=300)
    plt.close()

def train_surrogates(config_path):
    cfg = load_config(config_path)
    
    output_dir = "models"
    figures_dir = os.path.join("reports", "figures", "validation")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Training GPs (80/20 Split, No Scaling) ---")

    for exp in cfg['experiments']:
        exp_id = exp['id']
        print(f"\nProcessing: {exp_id} ({exp.get('title')})")
        
        try:
            df_X = pd.read_csv(exp['training_data']['input_path'])
            df_y = pd.read_csv(exp['training_data']['output_path'])
            
            # Save feature names for plotting
            feature_names = list(df_X.columns)
            
            # Select output column safely
            if "exp" in df_y.columns: y_vals = df_y["exp"].values
            elif "chi2" in df_y.columns: y_vals = df_y["chi2"].values
            else: y_vals = df_y.iloc[:, 0].values

            X = df_X.values
            y = y_vals.ravel()
            
        except Exception as e:
            print(f"!! Failed to load data: {e}")
            continue

        # Split Data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Kernel Selection
        if 'gp_override' in exp and 'kernel' in exp['gp_override']:
            k_name = exp['gp_override']['kernel']
            print(f"   Kernel: {k_name} (Override)")
        else:
            k_name = cfg.get('defaults', {}).get('gp', {}).get('kernel', 'RBF')
            print(f"   Kernel: {k_name} (Default)")

        kernel = get_kernel(k_name, n_features=X.shape[1])

        # Train
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True, 
            random_state=42
        )
        gp.fit(X_train, y_train)
        print(f"   Log Marginal Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.4f}")

        # Validate
        y_pred, y_std = gp.predict(X_val, return_std=True)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        within_2sigma = np.mean(np.abs(y_val - y_pred) <= 2 * y_std)
        print(f"   -> Val RMSE: {rmse:.5f} | Coverage: {within_2sigma*100:.1f}%")

        # --- PLOTTING ---
        # 1. Parity and Residuals (using Validation data)
        plot_diagnostics(y_val, y_pred, y_std, exp.get('title'), figures_dir, exp_id)
        
        # 2. Slice Plots (using Training data range + 1000 predicted points)
        plot_gp_slices(gp, X_train, y_train, feature_names, figures_dir, exp_id)

        # Save
        joblib.dump(gp, os.path.join(output_dir, f"{exp_id}_gp.joblib"))
        print(f"   -> Saved model and plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    train_surrogates(args.config)