import os
import joblib
import numpy as np
import emcee
import arviz as az
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class IntegralExperiment:
    """
    Evaluates the likelihood for an integral experiment using its specific GP.
    """
    def __init__(self, exp_id, gp_path, y_meas, y_err, weight=1.0):
        self.id = exp_id
        self.weight = 1.0
        self.y_meas = y_meas
        self.y_err = y_err
        
        # Load the surrogate model artifacts
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {exp_id} not found at {gp_path}")
            
        self.gp = joblib.load(gp_path)

    def get_log_likelihood(self, theta):
        """
        Calculates ln L(D_i | theta)
        """
        
        # 2. GP Prediction
        pred_mean, pred_std = self.gp.predict(theta.reshape(-1,1), return_std=True)
        
        # Flatten (handle cases where predict returns shape (1,1) or (1,))
        mu_gp = pred_mean.item()
        sigma_gp = pred_std.item()
        
        # 3. Combine Uncertainties (Experiment + Emulator)
        sigma_total = np.sqrt(self.y_err**2 + sigma_gp**2)
        
        # 4. Compute Log Likelihood (Gaussian)
        return norm.logpdf(self.y_meas, loc=mu_gp, scale=sigma_total)

class MicroscopicExperiment:
    """
    Evaluates the likelihood for a microscopic experiment using its specific GP.
    """
    def __init__(self, exp_id, C, gp_path, weight=1.0):
        self.id = exp_id
        self.C = C
        self.weight = weight
        self.gp = joblib.load(gp_path)

    def get_log_likelihood(self, theta):
        # GP predicts the Chi-Squared value
        chi2_val = self.gp.predict(theta.reshape(-1,1))
        chi2 = max(0.0, chi2_val.item()) # Clamp to 0
        
        # Standard Likelihood = -0.5 * Chi2
        ll = self.C - 0.5 * chi2
        
        # adjust ll with weight if needed
        return self.weight * ll

class JointPosterior:
    """
    The Master Class.
    Evaluates Sum(Log Priors) + Sum(Log Likelihoods of ALL experiments).
    """
    def __init__(self, prior_means, prior_stds, experiment_models):
        self.prior_means = np.array(prior_means)
        self.prior_cov = np.diag(np.array(prior_stds)**2)
        self.experiments = experiment_models

    def log_prior(self, theta):
        """
        Multivariate Gaussian Prior
        """
        return multivariate_normal.logpdf(theta, mean=self.prior_means, cov=self.prior_cov)

    def __call__(self, theta):
        """
        This is the function called by Emcee.
        Returns: ln P(theta | All Data)
        """
        # 1. Evaluate Prior
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        # 2. Sum Likelihoods across ALL experiments
        total_ll = 0.0
        for exp in self.experiments:
            ll = exp.get_log_likelihood(theta)
            total_ll += ll
            
        # 3. Return Joint Posterior
        return lp + total_ll

def run_joint_mcmc(config_path):
    # 1. Load Configuration
    cfg = load_config(config_path)
    print(f"--- Starting Joint Inference: {cfg['project_name']} ---")

    # 2. Parse Prior Information
    prior_names = cfg['parameters']['names']
    prior_means = cfg['parameters']['prior_means']
    prior_stds = cfg['parameters']['prior_stds']


    print(f"   Calibrating {len(prior_names)} parameters: {prior_names}")

    # 3. Initialize Evaluators for each Experiment
    models = []
    
    print("\n   Loading Experiment Models...")
    for exp in cfg['experiments']:
        gp_path = os.path.join("models", f"{exp['id']}_gp.joblib")
        scaler_path = os.path.join("models", f"{exp['id']}_scaler.joblib")
        
        try:
            if exp['type'] == "integral":
                exp_obj = IntegralExperiment(
                    exp_id=exp['id'],
                    gp_path=gp_path,
                    y_meas=exp['experimental_data']['measurement'],
                    y_err=exp['experimental_data']['uncertainty']
                )
            elif exp['type'] == "microscopic":
                exp_obj = MicroscopicExperiment(
                    exp_id=exp['id'],
                    C = exp['C_constant'],
                    gp_path=gp_path,
                )
            models.append(exp_obj)
            print(f"    - Loaded {exp['id']}")
        except FileNotFoundError:
            print(f"    !! Skipping {exp['id']}: GP Model not found.")

    if not models:
        print("No valid experiments found. Exiting.")
        return

    # 4. Construct the Joint Posterior
    posterior_fn = JointPosterior(prior_means, prior_stds, models)

    # 5. Setup MCMC (Emcee)
    n_walkers = cfg['defaults']['mcmc']['n_walkers']
    n_steps = cfg['defaults']['mcmc']['n_steps']
    ndim = len(prior_means)

    # Initialize walkers in a tight ball around the prior mean
    # (Standard practice: start near the prior center)
    pos = np.random.normal(
        loc=prior_means, 
        scale=np.array(prior_stds),
        size=(n_walkers, ndim)
    )

    filename = os.path.join(cfg['output_dir'], "joint_calibration_chain.h5")
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, ndim)

    print(f"\n   Running MCMC: {n_walkers} walkers x {n_steps} steps")
    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, posterior_fn)
    state_VC = sampler.run_mcmc(pos, n_steps, progress=True)
    
    # 6. Save Result as Arviz NetCDF
    print("\n   Saving results to NetCDF...")
    idata = az.from_emcee(sampler, var_names=prior_names)
    
    nc_path = os.path.join(cfg['output_dir'], "joint_posterior.nc")
    az.to_netcdf(idata, nc_path)
    
    # 7. Quick Diagnostics
    rhat = az.rhat(idata)
    print(f"   Max R-hat: {float(rhat.to_array().max()):.4f}")
    print(f"   Results saved to: {nc_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_joint_mcmc(args.config)