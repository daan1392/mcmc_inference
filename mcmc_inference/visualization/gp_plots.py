import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_parity(y_true, y_pred, y_std, title, save_path):
    """
    Plots Predicted vs True values with error bars.
    Ideal: All points lie on the diagonal line.
    """
    plt.figure(figsize=(6, 6))
    
    # Plot error bars (GP Uncertainty)
    plt.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='o', 
                 alpha=0.5, ecolor='gray', capsize=2, label='GP Prediction (95% CI)')
    
    # Plot Identity Line (Perfect Fit)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buff = (max_val - min_val) * 0.1
    lims = [min_val - buff, max_val + buff]
    
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Fit')
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True Simulation Output")
    plt.ylabel("GP Predicted Output")
    plt.title(f"Parity Plot: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_residuals(y_true, y_pred, y_std, title, save_path):
    """
    Plots Standardized Residuals (Z-scores).
    Ideal: Points scattered randomly around 0 between -3 and +3.
    """
    # Calculate Z-score: (True - Pred) / Std_Dev
    # Avoid division by zero
    residuals = (y_true - y_pred)
    z_scores = residuals / (y_std + 1e-9)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Residuals vs Index (Check for bias)
    ax1.scatter(range(len(z_scores)), z_scores, alpha=0.6)
    ax1.axhline(0, color='k', linestyle='--')
    ax1.axhline(3, color='r', linestyle=':', label='+/- 3 Sigma')
    ax1.axhline(-3, color='r', linestyle=':')
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Standardized Residual (Z-score)")
    ax1.set_title("Residuals vs Sample")
    ax1.legend()
    
    # Subplot 2: Histogram (Check for Normality)
    ax2.hist(z_scores, bins=20, density=True, alpha=0.6, color='green', label='Residuals')
    
    # Overlay Standard Normal Curve
    x = np.linspace(-5, 5, 100)
    ax2.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='N(0,1)')
    ax2.set_xlabel("Z-score")
    ax2.set_title("Residual Distribution")
    ax2.legend()
    
    plt.suptitle(f"Residual Diagnostics: {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def validate_models(config_path):
    cfg = load_config(config_path)
    
    # Create directory for reports
    report_dir = os.path.join("reports", "gp_performance")
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"--- Validating Surrogates for: {cfg['project_name']} ---")
    
    results = []

    for exp in cfg['experiments']:
        print(f"\nTesting: {exp['id']}")
        
        # 1. Load Data
        try:
            df_X = pd.read_csv(exp['training_data']['input_path'])
            df_y = pd.read_csv(exp['training_data']['output_path'])
            
            # Ensure 1D output
            X = df_X.values
            y_true = df_y.values[:,0].ravel()
            
        except FileNotFoundError:
            print(f"  !! Data not found for {exp['id']}. Skipping.")
            continue

        # 2. Load Model
        gp_path = os.path.join("models", f"{exp['id']}_gp.joblib")
        scaler_path = os.path.join("models", f"{exp['id']}_scaler.joblib")
        
        if not os.path.exists(gp_path):
            print(f"  !! Model not found at {gp_path}. Train first.")
            continue
            
        gp = joblib.load(gp_path)
        scaler = joblib.load(scaler_path)
        
        # 3. Predict (Goodness of Fit)
        # Scale inputs!
        X_scaled = scaler.transform(X)
        y_pred, y_std = gp.predict(X_scaled, return_std=True)
        y_pred = y_pred.ravel()
        y_std = y_std.ravel()
        
        # 4. Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Standardized Log Loss (Simpler check: % within 2 sigma)
        within_2sigma = np.mean(np.abs(y_true - y_pred) <= 2 * y_std)
        
        print(f"  -> RMSE: {rmse:.5f}")
        print(f"  -> R2 Score: {r2:.5f}")
        print(f"  -> Data within 2σ: {within_2sigma*100:.1f}% (Ideal ~95%)")
        
        results.append({
            "Experiment": exp['id'],
            "RMSE": rmse,
            "R2": r2,
            "Within_2Sigma": within_2sigma
        })
        
        # 5. Plot
        # Parity Plot
        plot_parity(
            y_true, y_pred, y_std, 
            title=exp['title'], 
            save_path=os.path.join(report_dir, f"{exp['id']}_parity.png")
        )
        
        # Residual Plot
        plot_residuals(
            y_true, y_pred, y_std, 
            title=exp['title'], 
            save_path=os.path.join(report_dir, f"{exp['id']}_residuals.png")
        )

    # Save summary CSV
    if results:
        pd.DataFrame(results).to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)
        print(f"\nValidation complete. plots saved to {report_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    validate_models(args.config)