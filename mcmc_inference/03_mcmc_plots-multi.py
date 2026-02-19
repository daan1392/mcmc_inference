import os
import argparse
import joblib
import arviz as az
import matplotlib.pyplot as plt
import yaml
import numpy as np

# 

# plt.rcParams.update(
#     {
#         "text.usetex": False,
#         "font.family": "STIXGeneral",
#         "mathtext.fontset": "cm",
#         "axes.formatter.use_mathtext": True,
#         "font.size": 16,
#     }
# )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class IntegralExperiment:
    def __init__(self, exp, gp_path):
        self.id = exp["id"]
        self.title = exp["title"]
        self.type = exp["type"]
        self.y_meas = exp["experimental_data"]["measurement"]
        self.y_err = exp["experimental_data"]["uncertainty"]
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {self.id} not found at {gp_path}")
        self.gp = joblib.load(gp_path)

    def get_chi2(self, y):
        return ((y - self.y_meas) / self.y_err) ** 2


class MicroscopicExperiment:
    def __init__(self, exp, gp_path):
        self.id = exp["id"]
        self.title = exp["title"]
        self.type = exp["type"]
        self.y_meas = 0.0
        self.y_err = 1.0
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {self.id} not found at {gp_path}")
        self.gp = joblib.load(gp_path)
    
class NormalizationFactor:
    """
    Evaluates the likelihood for a microscopic experiment using its specific GP.
    Now supports mapping specific global parameters to GP inputs.
    """

    def __init__(self, exp):
        self.id = exp["id"]
        self.title = "Normalization"
        self.type = exp["type"]
        self.unc = exp["experimental_data"]["uncertainty"]

    def get_chi2(self, y):
        return (1.0 - y) ** 2 / (self.unc ** 2)

def trace_plot(idata, project_name, save_path):
    axes = az.plot_trace(idata, compact=False)
    fig = axes[0, 0].figure
    fig.suptitle(f"Trace Plot: {project_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


def corner_plot(idata, save_path):
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)

    if n_params > 1:
        az.plot_pair(
            idata,
            kind="kde",
            marginals=True,
            point_estimate="median",
            textsize=12,
            kde_kwargs={"fill_last": False, "contourf_kwargs": {"cmap": "viridis"}},
        )
    else:
        az.plot_posterior(idata, point_estimate="mean", hdi_prob=0.95, textsize=12)

    plt.savefig(save_path, dpi=300)
    plt.close()


def forest_plot(idata, save_path):
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)
    fig_height = max(4, n_params * 0.5 + 2)
    az.plot_forest(idata, combined=True, hdi_prob=0.95, figsize=(8, fig_height))
    plt.title("Posterior 95% HDI")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def input_pdf_plot(prior_samples, posterior_samples, param_names, save_path):
    """
    Plots the prior and posterior predictive distributions for inputs.
    Handles multiple dimensions by creating subplots.
    """
    n_params = prior_samples.shape[1]
    
    # Determine grid layout
    cols = min(n_params, 3)
    rows = int(np.ceil(n_params / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i in range(n_params):
        ax = axs[i]
        # Slice 1D array for histogram to avoid "multiple dataset" error
        ax.hist(prior_samples[:, i], bins=30, density=True, alpha=0.5, label="Prior", color="C1")
        ax.hist(posterior_samples[:, i], bins=30, density=True, alpha=0.5, label="Posterior", color="C2")
        
        ax.set_title(f"{param_names[i]}")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    fig.suptitle("Input Parameter Distributions: Prior vs Posterior")
    fig.savefig(save_path, dpi=300)
    plt.close()


def output_scatter_plot(exp, prior, posterior, param_idx, param_name, save_path=None, axs=None):
    """
    Plots output vs ONE specific input parameter.
    """
    if axs is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    else:
        ax = axs

    # 1. Predictions use the FULL parameter set (N, 5)
    y_prior_pred = exp.gp.predict(prior)
    y_post_pred = exp.gp.predict(posterior)

    # 2. X-axis uses ONLY the current parameter column (N, 1)
    x_prior = prior[:, param_idx]
    x_post = posterior[:, param_idx]

    if exp.type == "integral":
        ax.axhline(exp.y_meas, ls="--", label="Measurement", color="C0")
        ax.axhspan(
            exp.y_meas - 2 * exp.y_err,
            exp.y_meas + 2 * exp.y_err,
            facecolor="C0", alpha=0.2,
        )

    ax.plot(x_prior, y_prior_pred, color="C1", alpha=0.5, ls="None", marker=".", label="Prior")
    ax.plot(x_post, y_post_pred, color="C2", ls="None", alpha=0.5, marker=".", label="Posterior")

    ax.set(
        title=f"{exp.id}",
        xlabel=f"{param_name}",
        ylabel=r"$k_{\text{eff}}$" if exp.type =="microscopic" else r"$\chi^2$",
    )
    # Only add legend if it's a standalone plot
    if axs is None:
        ax.legend()
        fig.savefig(save_path, dpi=300)
        plt.close()

def plot_chi2(exp, prior, posterior, param_idx, param_name, save_path=None, axs=None):
    """
    Plots output vs ONE specific input parameter.
    """
    if axs is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    else:
        ax = axs

    # 1. Predictions use the FULL parameter set (N, 5)
    y_prior_pred = exp.gp.predict(prior) if exp.type !="normalization" else prior[:, -1]
    y_post_pred = exp.gp.predict(posterior) if exp.type !="normalization" else posterior[:, -1]

    # 2. X-axis uses ONLY the current parameter column (N, 1)
    x_prior = prior[:, param_idx]
    x_post = posterior[:, param_idx]

    ax.plot(x_prior, y_prior_pred if exp.type =="microscopic" else exp.get_chi2(y_prior_pred), color="C1", alpha=0.5, ls="None", marker=".", label="Prior")
    ax.plot(x_post, y_post_pred if exp.type =="microscopic" else exp.get_chi2(y_post_pred), color="C2", ls="None", alpha=0.5, marker=".", label="Posterior")

    ax.set(
        # title=f"{exp.title}",
        xlabel=f"{param_name}",
        ylabel=r"$\chi^2$",
    )

    ax.spines[["right", "top"]].set_visible(False)

    if axs is None:
        ax.legend()
        fig.savefig(save_path, dpi=300)
        plt.close()


def output_scatterpdf_plot(exp, prior, posterior, param_idx, param_name, save_path):
    """
    Plots Scatter + Marginals for ONE specific input parameter vs Output.
    """
    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        figsize=(5, 5),
        width_ratios=(6, 1),
        height_ratios=(1, 6),
        layout="constrained",
    )

    ax = axs["scatter"]
    axs["histx"].tick_params(axis="x", labelbottom=False, bottom=False)
    axs["histx"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="x", labelbottom=False, bottom=False)

    # --- Data Preparation ---
    # GP needs full shape (N, D)
    y_prior_pred = exp.gp.predict(prior)
    y_post_pred = exp.gp.predict(posterior)
    
    # Plotting needs specific column (N,)
    x_prior = prior[:, param_idx]
    x_post = posterior[:, param_idx]

    # --- Plotting ---
    if exp.type == "integral":
        ax.axhline(exp.y_meas, ls="--", label="Measurement", color="C0")
        ax.axhspan(
            exp.y_meas - 2 * exp.y_err,
            exp.y_meas + 2 * exp.y_err,
            facecolor="C0", alpha=0.2,
        )

    # Scatter
    ax.plot(x_prior, y_prior_pred, color="C1", alpha=0.3, ls="None", marker=".", label="Prior")
    
    # Hist X (Top) - Input Parameter Distribution
    axs["histx"].hist(x_prior, bins=30, density=True, alpha=0.5, color="C1")

    # Hist Y (Right) - Output Distribution
    axs["histy"].hist(y_prior_pred, bins=30, density=True, alpha=0.5, color="C1", orientation="horizontal")

    # Scatter Posterior
    ax.plot(x_post, y_post_pred, color="C2", ls="None", alpha=0.5, marker=".", label="Posterior")
    
    # Hist X (Top) Posterior
    weight = np.ones_like(x_post) / len(x_post) # Simple weighting, or density=True handles it
    axs["histx"].hist(x_post, bins=30, density=True, alpha=0.5, color="C2")

    # Hist Y (Right) Posterior
    axs["histy"].hist(y_post_pred, bins=30, density=True, alpha=0.5, color="C2", orientation="horizontal")

    ax.set(
        xlabel=f"{param_name}",
        ylabel=r"$k_{\text{eff}}$" if exp.type == "integral" else r"$\chi^2$",
    )
    ax.legend(loc='upper left', fontsize=10)

    # Sync limits
    axs["histx"].set(xlim=ax.get_xlim())
    axs["histy"].set(ylim=ax.get_ylim())

    # Remove spines for histograms
    ax.spines[["right", "top"]].set_visible(False)
    axs["histx"].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axs["histy"].spines[["left", "bottom", "right", "top"]].set_visible(False)

    fig.savefig(save_path, dpi=300)
    plt.close()


def plot_chi2pdf(exp, prior, posterior, param_idx, param_name, save_path):
    """
    Plots Scatter + Marginals for ONE specific input parameter vs Output.
    """

    
    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        figsize=(5, 5),
        width_ratios=(6, 1),
        height_ratios=(1, 6),
        layout="constrained",
    )

    ax = axs["scatter"]
    axs["histx"].tick_params(axis="x", labelbottom=False, bottom=False)
    axs["histx"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="x", labelbottom=False, bottom=False)

    # --- Data Preparation ---
    y_prior_pred = exp.gp.predict(prior) if exp.type != "normalization" else prior[:, param_idx]
    y_post_pred = exp.gp.predict(posterior) if exp.type != "normalization" else posterior[:, param_idx]
    
    # Plotting needs specific column (N,)
    x_prior = prior[:, param_idx]
    x_post = posterior[:, param_idx]

    # # --- Plotting ---
    # if exp.type == "integral":
    #     ax.axhline(exp.y_meas, ls="--", label="Measurement", color="C0")
    #     ax.axhspan(
    #         exp.y_meas - 2 * exp.y_err,
    #         exp.y_meas + 2 * exp.y_err,
    #         facecolor="C0", alpha=0.2,
    #     )

    # Scatter
    ax.plot(x_prior, y_prior_pred if exp.type =="microscopic" else exp.get_chi2(y_prior_pred), color="C1", alpha=0.3, ls="None", marker=".", label="Prior")
    
    # Hist X (Top) - Input Parameter Distribution
    axs["histx"].hist(x_prior, bins=30, density=True, alpha=0.5, color="C1")

    # Hist Y (Right) - Output Distribution
    axs["histy"].hist(y_prior_pred if exp.type =="microscopic" else exp.get_chi2(y_prior_pred), bins=30, density=True, alpha=0.5, color="C1", orientation="horizontal")

    # Scatter Posterior
    ax.plot(x_post, y_post_pred if exp.type =="microscopic" else exp.get_chi2(y_post_pred), color="C2", ls="None", alpha=0.5, marker=".", label="Posterior")
    
    # Hist X (Top) Posterior
    weight = np.ones_like(x_post) / len(x_post) # Simple weighting, or density=True handles it
    axs["histx"].hist(x_post, bins=30, density=True, alpha=0.5, color="C2")

    # Hist Y (Right) Posterior
    axs["histy"].hist(y_post_pred if exp.type =="microscopic" else exp.get_chi2(y_post_pred), bins=30, density=True, alpha=0.5, color="C2", orientation="horizontal")

    ax.set(
        xlabel=f"{param_name}",
        ylabel=r"$\chi^2$",
    )
    ax.legend(loc='upper left', fontsize=10)

    # Sync limits
    axs["histx"].set(xlim=ax.get_xlim())
    axs["histy"].set(ylim=ax.get_ylim())

    # Remove spines for histograms
    ax.spines[["right", "top"]].set_visible(False)
    axs["histx"].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axs["histy"].spines[["left", "bottom", "right", "top"]].set_visible(False)

    fig.savefig(save_path, dpi=300)
    plt.close()


def plot_mcmc_results(config_path):
    # 1. Load Configuration
    cfg = load_config(config_path)

    # Define paths
    netcdf_path = os.path.join(cfg["output_dir"], "joint_posterior.nc")
    figures_dir = cfg["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    print(f"--- Loading Results from: {netcdf_path} ---")

    if not os.path.exists(netcdf_path):
        print(f"!! Error: File not found. Run inference first.")
        return

    # 2. Load Data
    idata = az.from_netcdf(netcdf_path)
    param_names = cfg["parameters"]["names"]
    param_labels = cfg["parameters"]["labels"]
    n_params = len(param_names)
    print(f"Parameters found: {param_names} ({n_params} total)")

    # 3. Initialize evaluators
    models = []
    print("\n   Loading Experiment Models...")
    for exp in cfg["experiments"]:
        gp_path = os.path.join("models", f"{exp['id']}_gp.joblib")
        try:
            if exp["type"] == "integral":
                exp_obj = IntegralExperiment(
                    exp=exp,
                    gp_path=gp_path,
                )
            elif exp["type"] == "microscopic":
                exp_obj = MicroscopicExperiment(
                    exp=exp,
                    gp_path=gp_path,
                )
            elif exp["type"] == "normalization":
                exp_obj = NormalizationFactor(
                    exp=exp,
                )
            models.append(exp_obj)
            print(f"    - Loaded {exp['id']}")
        except FileNotFoundError:
            print(f"    !! Skipping {exp['id']}: GP Model not found.")

    # Generate samples
    n_prior_samples = 10000
    prior_means = np.array(cfg["parameters"]["prior_means"])
    prior_stds = np.array(cfg["parameters"]["prior_stds"]) * prior_means
    prior_X_samples = np.random.normal(
        loc=prior_means, scale=prior_stds, size=(n_prior_samples, n_params)
    )

    burnin = cfg["defaults"]["mcmc"]["burn_in"]
    posterior_sel = idata.posterior.isel(draw=slice(burnin, None))
    posterior_stacked = posterior_sel.stack(sample=("chain", "draw"))
    
    # Ensure correct column order matching param_names
    param_arrays = [posterior_stacked[param].values.reshape(-1) for param in param_names]
    posterior_X_samples = np.vstack(param_arrays).T

    # --- Plots ---
    
    # print("Generating Trace Plot...")
    # trace_plot(idata, cfg["project_name"], os.path.join(figures_dir, "trace_plot.png"))

    # print("Generating Corner Plot...")
    # corner_plot(idata, os.path.join(figures_dir, "corner_plot.png"))

    # print("Generating Forest Plot...")
    # forest_plot(idata, os.path.join(figures_dir, "forest_plot.png"))

    # print("Generating Input Plots...")
    # # Updated to handle multi-dim
    # input_pdf_plot(
    #     prior_X_samples, 
    #     posterior_X_samples, 
    #     param_names, 
    #     os.path.join(figures_dir, "input_pdf_plot.png")
    # )

    # print("Generating Output Plots...")
    # for exp in models:
    #     # Loop over parameters to create one scatter plot per input dimension
    #     for i, p_name in enumerate(param_names):
    #         clean_p_name = p_name.replace(" ", "_").replace("/", "_")
    #         plot_chi2pdf(
    #             exp,
    #             prior_X_samples,
    #             posterior_X_samples,
    #             param_idx=i,
    #             param_name=param_labels[i],
    #             save_path=os.path.join(figures_dir, f"{exp.id}_chi2_{clean_p_name}.png"),
    #         )

    print("Generating Combined Output Plot...")
    # Create a grid: Rows = Experiments, Cols = Parameters
    fig, axs = plt.subplots(
        len(models), n_params, 
        figsize=(4 * n_params, 4 * len(models)), 
        constrained_layout=True
    )

    # Ensure axs is always 2D
    if len(models) == 1 and n_params == 1:
        axs = np.array([[axs]])
    elif len(models) == 1:
        axs = axs[np.newaxis, :]
    elif n_params == 1:
        axs = axs[:, np.newaxis]

    for row, exp in enumerate(models):
        for col, p_name in enumerate(param_names):
            if col == 2:
                axs[row,col].set(title=exp.title)
            plot_chi2(
                exp,
                prior_X_samples,
                posterior_X_samples,
                param_idx=col,
                param_name=param_labels[col],
                save_path=os.path.join(figures_dir, f"{exp.id}_chi2_{p_name}.png"),
                axs=axs[row, col]
            )

    # bbox_inches='tight' is crucial here so the top row's title isn't cut off
    fig.savefig(f"{figures_dir}/combined_output_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Summary ---
    print("Generating Summary Table...")
    summary_df = az.summary(posterior_sel, hdi_prob=0.95)
    summary_path = os.path.join(figures_dir, "posterior_summary.csv")
    summary_df.to_csv(summary_path)

    print(f"Full summary saved to {summary_path}")

    # --- Generate CSV and Markdown Summaries ---
    print("Generating Summary Tables...")
    summary_df = az.summary(posterior_sel, hdi_prob=0.95)
    
    # Save CSV
    csv_path = os.path.join(figures_dir, "posterior_summary.csv")
    summary_df.to_csv(csv_path)

    # Save Markdown Report
    md_path = os.path.join(figures_dir, "summary_report.md")
    
    with open(md_path, "w") as f:
        f.write(f"# MCMC Inference Summary: {cfg.get('project_name', 'Results')}\n\n")
        f.write("## Parameter Statistics\n\n")
        f.write("| Parameter | Prior Mean | Rel. Prior Std (Abs) | Posterior Mean | Rel. Posterior Std | Change in Mean (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for i, param in enumerate(param_names):
            # Prior stats from config/calculation
            pr_mu = prior_means[i]
            pr_std = prior_stds[i] 
            
            try:
                po_mu = summary_df.loc[param, "mean"]
                po_std = summary_df.loc[param, "sd"] / summary_df.loc[param, "mean"]
                
                # Calculate % change
                pct_change = ((po_mu - pr_mu) / pr_mu) * 100
                
                f.write(f"| {param} | {pr_mu:.4e} | {(pr_std/pr_mu):.4e} | {po_mu:.4e} | {po_std:.4e} | {pct_change:+.2f}% |\n")
            except KeyError:
                print(f"Warning: Parameter {param} found in priors but missing in posterior summary.")

        f.write("\n\n## Calibration Metrics\n\n")
        f.write("| Parameter | R_hat | ESS (Bulk) | ESS (Tail) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for param in param_names:
            try:
                r_hat = summary_df.loc[param, "r_hat"]
                ess_bulk = summary_df.loc[param, "ess_bulk"]
                ess_tail = summary_df.loc[param, "ess_tail"]
                f.write(f"| {param} | {r_hat:.3f} | {ess_bulk:.1f} | {ess_tail:.1f} |\n")
            except KeyError:
                pass

    print(f"Full summary saved to: \n  - {csv_path}\n  - {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_multi.yaml")
    args = parser.parse_args()

    plot_mcmc_results(args.config)