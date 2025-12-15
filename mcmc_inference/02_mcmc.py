import os
import joblib
import numpy as np
import emcee
import arviz as az
from scipy.stats import multivariate_normal, norm
import yaml
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class IntegralExperiment:
    """
    Evaluates the likelihood for an integral experiment using its specific GP.
    Now supports mapping specific global parameters to GP inputs.
    """

    def __init__(self, exp_id, gp_path, y_meas, y_err, input_indices, weight=1.0):
        self.id = exp_id
        self.weight = weight
        self.y_meas = y_meas
        self.y_err = y_err
        self.input_indices = input_indices

        # Load the surrogate model artifacts
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {exp_id} not found at {gp_path}")

        self.gp = joblib.load(gp_path)

    def get_log_likelihood(self, theta):
        """
        Calculates ln L(D_i | theta)
        """
        relevant_theta = theta[self.input_indices]

        pred_mean, pred_std = self.gp.predict(relevant_theta.reshape(1, -1), return_std=True)

        sigma_total = np.sqrt(self.y_err**2) + pred_std.item()**2)

        return norm.logpdf(self.y_meas, loc=pred_mean.item(), scale=sigma_total)


class MicroscopicExperiment:
    """
    Evaluates the likelihood for a microscopic experiment using its specific GP.
    Now supports mapping specific global parameters to GP inputs.
    """

    def __init__(self, exp_id, C, gp_path, input_indices, weight=1.0):
        self.id = exp_id
        self.C = C
        self.weight = weight
        self.input_indices = input_indices 
        self.gp = joblib.load(gp_path)

    def get_log_likelihood(self, theta):
        relevant_theta = theta[self.input_indices]

        chi2_val = self.gp.predict(relevant_theta.reshape(1, -1))

        chi2 = max(0.0, chi2_val.item())

        ll = self.C - 0.5 * chi2

        return self.weight * ll


class JointPosterior:
    """
    The Master Class.
    Evaluates Sum(Log Priors) + Sum(Log Likelihoods of ALL experiments).
    """

    def __init__(self, prior_means, prior_stds, experiment_models):
        self.prior_means = np.array(prior_means)
        self.prior_cov = np.diag(np.array(prior_stds) ** 2)
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
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        total_ll = 0.0
        for exp in self.experiments:
            ll = exp.get_log_likelihood(theta)
            total_ll += ll

        return lp + total_ll


def run_joint_mcmc(config_path):
    # 1. Load configuration
    cfg = load_config(config_path)
    print(f"--- Starting Joint Inference: {cfg['project_name']} ---")

    # 2. Parse prior information
    global_param_names = cfg["parameters"]["names"]
    prior_means = np.array(cfg["parameters"]["prior_means"])
    print(prior_means)
    prior_stds = np.array(cfg["parameters"]["prior_stds"]) * prior_means

    print(f"   Calibrating {len(global_param_names)} parameters: {global_param_names}")

    # 3. Initialize each experiment
    models = []

    print("\n   Loading Experiment Models...")
    for exp in cfg["experiments"]:
        gp_path = os.path.join("models", f"{exp['id']}_gp.joblib")

        exp_param_names = exp.get("parameters", global_param_names)

        try:
            input_indices = [global_param_names.index(name) for name in exp_param_names]
        except ValueError as e:
            print(f"Error in {exp['id']}: Parameter name mismatch. {e}")
            continue

        try:
            if exp["type"] == "integral":
                exp_obj = IntegralExperiment(
                    exp_id=exp["id"],
                    gp_path=gp_path,
                    y_meas=exp["experimental_data"]["measurement"],
                    y_err=exp["experimental_data"]["uncertainty"],
                    input_indices=input_indices,
                )
            elif exp["type"] == "microscopic":
                exp_obj = MicroscopicExperiment(
                    exp_id=exp["id"],
                    C=exp["training_data"]["C_constant"],
                    gp_path=gp_path,
                    input_indices=input_indices,
                )
            models.append(exp_obj)
            print(
                f"    - Loaded {exp['id']} (Inputs: {exp_param_names} -> Indices: {input_indices})"
            )
        except FileNotFoundError:
            print(f"    !! Skipping {exp['id']}: GP Model not found.")

    if not models:
        print("No valid experiments found. Exiting.")
        return

    # 4. Calculate the joint posterior
    posterior_fn = JointPosterior(prior_means, prior_stds, models)

    # 5. Setup MCMC (Emcee)
    n_walkers = cfg["defaults"]["mcmc"]["n_walkers"]
    n_steps = cfg["defaults"]["mcmc"]["n_steps"]
    ndim = len(prior_means)

    pos = np.random.normal(
        loc=prior_means,
        scale=np.array(prior_stds) * 0.1,
        size=(n_walkers, ndim),
    )

    filename = os.path.join(cfg["output_dir"], "joint_calibration_chain.h5")
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, ndim)

    print(f"\n   Running MCMC: {n_walkers} walkers x {n_steps} steps")

    sampler = emcee.EnsembleSampler(n_walkers, ndim, posterior_fn, backend=backend)
    state_VC = sampler.run_mcmc(pos, n_steps, progress=True)

    # 6. Save Result as Arviz NetCDF
    print("\n   Saving results to NetCDF...")
    idata = az.from_emcee(sampler, var_names=global_param_names)

    nc_path = os.path.join(cfg["output_dir"], "joint_posterior.nc")
    az.to_netcdf(idata, nc_path)

    # 7. Print summary statistics
    summary = az.summary(idata, round_to=4)
    rhat = az.rhat(idata)
    print(f"   Max R-hat: {float(rhat.to_array().max()):.4f}")
    print(f"   Results saved to: {nc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    run_joint_mcmc(args.config)
