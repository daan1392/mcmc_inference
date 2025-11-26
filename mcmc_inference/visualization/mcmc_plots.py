import os
import argparse
import joblib
import arviz as az
import matplotlib.pyplot as plt
import yaml
import numpy as np

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def trace_plot(idata, project_name, save_path):
    # 'compact=False' separates variables into different subplots
    axes = az.plot_trace(idata, compact=False)
    
    # Adjust layout and save
    fig = axes[0,0].figure
    fig.suptitle(f"Trace Plot: {project_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def corner_plot(idata, save_path):
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)
    
    if n_params > 1:
        # Corner plot makes sense only if we have correlations to show
        az.plot_pair(
            idata,
            kind='kde',
            marginals=True,
            point_estimate='median',
            textsize=12,
            kde_kwargs={'fill_last': False, 'contourf_kwargs': {'cmap': 'viridis'}}
        )
    else:
        # For 1 param, just show the distribution with Mean and 95% HDI
        az.plot_posterior(
            idata,
            point_estimate='mean',
            hdi_prob=0.95,
            textsize=12
        )
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def forest_plot(idata, save_path):
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)
    
    # Adjust height dynamically based on number of params so it doesn't look squashed
    fig_height = max(4, n_params * 0.5 + 2)
    
    az.plot_forest(
        idata, 
        combined=True, 
        hdi_prob=0.95, 
        figsize=(8, fig_height)
    )
    plt.title("Posterior 95% HDI")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def input_pdf_plot(prior_samples, posterior_samples, save_path):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        prior_samples,
        bins=30,
        density=True,
        alpha=0.5,
        label='Prior Samples'
    )
    ax.hist(
        posterior_samples,
        bins=30,
        density=True,
        alpha=0.5,
        label='Posterior Samples'
    )

    ax.set(
        title=f"Prior vs Posterior Predictive",
        xlabel="Response Variable",
        ylabel="Density"
    )
    ax.legend()
    fig.savefig(save_path, dpi=300)

def output_scatter_plot(exp_mean, exp_unc, prior_X_samples, prior_samples, posterior_X_samples, posterior_samples, save_path):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(
        exp_mean,
        ls='--',
        label='Measurement Samples'
    )

    ax.axhspan(exp_mean-2*exp_unc, exp_mean+2*exp_unc, 
           facecolor='C0',
           alpha=0.2,
           label='± 2σ Region')

    ax.plot(
        prior_X_samples,
        prior_samples,
        # bins=30,
        # density=True,
        alpha=0.5,
        ls='None',
        marker='.',
        label='Prior Samples'
    )
    ax.plot(
        posterior_X_samples,
        posterior_samples,
        # bins=30,
        # density=True,
        ls='None',
        alpha=0.5,
        marker='.',
        label='Posterior Samples'
    )

    ax.set(
        title=f"Prior vs Posterior Predictive",
        xlabel="Response Variable",
        ylabel="Density",
        # xlim=(0.9,1.1)
    )
    ax.legend()
    fig.savefig(save_path, dpi=300)

def output_pdf_plot(experiment_samples, prior_X_samples, prior_samples, posterior_X_samples, posterior_samples, save_path):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        experiment_samples,
        # bins=30,
        # density=True,
        alpha=0.5,
        label='Measurement Samples'
    )
    
    ax.hist(
        prior_samples,
        bins=30,
        density=True,
        alpha=0.5,
        label='Prior Samples'
    )
    ax.hist(
        posterior_samples,
        bins=30,
        density=True,
        alpha=0.5,
        label='Posterior Samples'
    )

    ax.set(
        title=f"Prior vs Posterior Predictive",
        xlabel="Response Variable",
        ylabel="Density",
        # xlim=(0.9,1.1)
    )
    ax.legend()
    fig.savefig(save_path, dpi=300)

def plot_mcmc_results(config_path):
    # 1. Load Configuration
    cfg = load_config(config_path)
    
    # Define paths
    netcdf_path = os.path.join(cfg['output_dir'], "joint_posterior.nc")
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"--- Loading Results from: {netcdf_path} ---")
    
    if not os.path.exists(netcdf_path):
        print(f"!! Error: File not found. Run inference first.")
        return

    # 2. Load Data (Arviz reads the NetCDF we saved earlier)
    idata = az.from_netcdf(netcdf_path)
    
    # Get parameter names for clearer logging
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)
    print(f"Parameters found: {param_names}")

    # 3. Prepare Data for Plots
    # Generate samples drawn from experimental mean and uncertainty assuming Gaussian
    n_exp_samples = 10000
    exp_samples = {}
    for exp in cfg['experiments']:
        exp_id = exp['id']
        y_meas = np.array(exp['experimental_data']['measurement']) if exp['type'] == 'integral' else 0
        y_err = np.array(exp['experimental_data']['uncertainty']) if exp['type'] == 'integral' else 1
        
        samples = np.random.normal(loc=y_meas, scale=y_err, size=(n_exp_samples, 1))
        exp_samples[exp_id] = samples

    # Generate prior samples for predictive checks
    n_prior_samples = 10000
    prior_means = cfg['parameters']['prior_means']
    prior_stds = cfg['parameters']['prior_stds']
    prior_X_samples = np.random.normal(loc=prior_means, scale=prior_stds, size=(n_prior_samples, n_params))
    
    # Extract posterior samples
    burnin = cfg['defaults']['mcmc']['burn_in']

    # select draws after burnin and stack chain/draw into a single sample axis
    posterior_sel = idata.posterior.isel(draw=slice(burnin, None))
    posterior_stacked = posterior_sel.stack(sample=("chain", "draw"))

    # collect parameter columns and build a (nsamples, n_params) array
    param_arrays = [posterior_stacked[param].values.reshape(-1) for param in param_names]
    posterior_all = np.vstack(param_arrays).T  # shape: (nsamples_total, n_params)

    # match the number of posterior samples to prior samples (n_prior_samples)
    n_post_needed = prior_X_samples.shape[0]
    if posterior_all.shape[0] >= n_post_needed:
        idx = np.random.choice(posterior_all.shape[0], size=n_post_needed, replace=False)
    else:
        idx = np.random.choice(posterior_all.shape[0], size=n_post_needed, replace=True)
    posterior_X_samples = posterior_all[idx]

    # Predict output responses for prior and posterior samples from GPs
    output_responses = {}
    for exp in cfg['experiments']:
        exp_id = exp['id']
        gp_path = os.path.join("models", f"{exp_id}_gp.joblib")
        scaler_path = os.path.join("models", f"{exp_id}_scaler.joblib")
        
        try:
            gp = joblib.load(gp_path)
            
            # Prior Predictions
            prior_preds = gp.predict(prior_X_samples)

            # Posterior Predictions
            post_preds = gp.predict(posterior_X_samples)
            
            output_responses[exp_id] = {
                'prior': prior_preds,
                'posterior': post_preds
            }
        except FileNotFoundError:
            print(f"    !! Skipping predictions for {exp_id}: GP Model not found.")
            continue
    
    # ---------------------------------------------------------
    # Plot 1: Trace Plot (Convergence Check)
    # ---------------------------------------------------------
    print("Generating Trace Plot...")
    trace_plot(idata, cfg['project_name'], os.path.join(figures_dir, "trace_plot.png"))

    # ---------------------------------------------------------
    # Plot 2: Corner Plot (Posterior Correlations)
    # ---------------------------------------------------------
    print("Generating Corner Plot...")
    corner_plot(idata, os.path.join(figures_dir, "corner_plot.png"))

    # ---------------------------------------------------------
    # Plot 3: Forest Plot (Summary of intervals)
    # ---------------------------------------------------------
    print("Generating Forest Plot...")
    forest_plot(idata, os.path.join(figures_dir, "forest_plot.png"))

    # ---------------------------------------------------------
    # 4. Plot prior and posterior input responses
    # ---------------------------------------------------------
    print("Generating Input Plots...")
    input_pdf_plot(prior_X_samples, posterior_X_samples, os.path.join(figures_dir, "input_pdf_plot.png"))

    # ---------------------------------------------------------
    # 4. Plot prior and posterior output responses
    # ---------------------------------------------------------
    print("Generating Output Plots...")
    for exp in cfg['experiments']:
        exp_id = exp['id']
        
        if exp['type'] == 'integral':
            output_scatter_plot(
                exp['experimental_data']['measurement'], 
                exp['experimental_data']['uncertainty'], 
                prior_X_samples, 
                output_responses[exp_id]['prior'], 
                posterior_X_samples, 
                output_responses[exp_id]['posterior'], 
                os.path.join(figures_dir, f"{exp_id}_output_pdf_plot.png")
            )
        else:
            output_scatter_plot(
                0, 
                1, 
                prior_X_samples, 
                output_responses[exp_id]['prior'], 
                posterior_X_samples, 
                output_responses[exp_id]['posterior'], 
                os.path.join(figures_dir, f"{exp_id}_output_pdf_plot.png")
            )



    # ---------------------------------------------------------
    # 5. Save Text Summary
    # ---------------------------------------------------------
    print("Generating Summary Table...")
    # This calculates Mean, SD, hdi_3%, hdi_97%, and R_hat
    summary_df = az.summary(idata, hdi_prob=0.95)
    
    summary_path = os.path.join("reports", "posterior_summary.csv")
    summary_df.to_csv(summary_path)
    
    print("\n--- Calibration Results ---")
    print(summary_df[['mean', 'sd', 'r_hat']])
    print(f"Full summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    plot_mcmc_results(args.config)