import os
import argparse
import joblib
import arviz as az
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 16,
    }
)

import yaml
import numpy as np


def load_config(config_path):
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class IntegralExperiment:
    """
    Evaluates the likelihood for an integral experiment using its specific GP.
    """

    def __init__(self, exp_id, exp_type, gp_path, y_meas, y_err):
        self.id = exp_id
        self.type = exp_type
        self.y_meas = y_meas
        self.y_err = y_err
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {exp_id} not found at {gp_path}")
        self.gp = joblib.load(gp_path)


class MicroscopicExperiment:
    """
    Evaluates the likelihood for a microscopic experiment using its specific GP.
    """

    def __init__(self, exp_id, exp_type, gp_path):
        self.id = exp_id
        self.type = exp_type
        self.y_meas = 0.0
        self.y_err = 1.0
        if not os.path.exists(gp_path):
            raise FileNotFoundError(f"GP model for {exp_id} not found at {gp_path}")
        self.gp = joblib.load(gp_path)


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
        # Corner plot makes sense only if we have correlations to show
        az.plot_pair(
            idata,
            kind="kde",
            marginals=True,
            point_estimate="median",
            textsize=12,
            kde_kwargs={"fill_last": False, "contourf_kwargs": {"cmap": "viridis"}},
        )
    else:
        # For 1 param, just show the distribution with Mean and 95% HDI
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


def input_pdf_plot(prior_samples, posterior_samples, save_path):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(prior_samples, bins=30, density=True, alpha=0.5, label="Prior Samples")
    ax.hist(posterior_samples, bins=30, density=True, alpha=0.5, label="Posterior Samples")

    ax.set(title=f"Prior vs Posterior Predictive", xlabel=r"$\Gamma_\gamma$, eV", ylabel="Density")
    ax.legend()
    fig.savefig(save_path, dpi=300)


def output_scatter_plot(exp, prior, posterior, save_path, axs=None):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    if axs is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    else:
        ax = axs

    if exp.type == "integral":
        ax.axhline(
            exp.y_meas,
            ls="--",
            label="Measurement",
            color="C0",
        )

        ax.axhspan(
            exp.y_meas - 2 * exp.y_err,
            exp.y_meas + 2 * exp.y_err,
            facecolor="C0",
            alpha=0.2,
        )

    ax.plot(
        prior,
        exp.gp.predict(prior),
        color="C1",
        alpha=0.5,
        ls="None",
        marker=".",
        label="Prior Samples",
    )
    ax.plot(
        posterior,
        exp.gp.predict(posterior),
        color="C2",
        ls="None",
        alpha=0.5,
        marker=".",
        label="Posterior Samples",
    )

    ax.set(
        title=f"{exp.id}",
        xlabel=r"$\Gamma_\gamma(E=4 keV)$, eV",
        ylabel=r"$k_{\text{eff}}$" if exp.type == "integral" else r"$\chi^2$",
        # ylim=(0,None) if exp.type == 'microscopic' else None,
    )
    ax.legend()

    if axs is None:
        fig.savefig(save_path, dpi=300)


def output_scatterpdf_plot(exp, prior, posterior, save_path):
    """
    Plots the prior and posterior predictive distributions for a given experiment.
    """
    fig, axs = plt.subplot_mosaic(
        [["histx", "."], ["scatter", "histy"]],
        figsize=(4, 4),
        width_ratios=(6, 1),
        height_ratios=(1, 6),
        layout="constrained",
    )

    ax = axs["scatter"]

    axs["histx"].tick_params(axis="x", labelbottom=False, bottom=False)
    axs["histx"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="y", labelleft=False, left=False)
    axs["histy"].tick_params(axis="x", labelbottom=False, bottom=False)

    if exp.type == "integral":
        ax.axhline(
            exp.y_meas,
            ls="--",
            label="Measurement",
            color="C0",
        )

        ax.axhspan(
            exp.y_meas - 2 * exp.y_err,
            exp.y_meas + 2 * exp.y_err,
            facecolor="C0",
            alpha=0.2,
        )

    ax.plot(
        prior,
        exp.gp.predict(prior),
        color="C1",
        alpha=0.5,
        ls="None",
        marker=".",
        label="Prior Samples",
    )

    axs["histx"].hist(
        prior,
        bins=30,
        density=True,
        alpha=0.5,
        color="C1",
    )

    axs["histy"].hist(
        exp.gp.predict(prior),
        bins=30,
        density=True,
        alpha=0.5,
        color="C1",
        orientation="horizontal",
    )

    ax.plot(
        posterior,
        exp.gp.predict(posterior),
        color="C2",
        ls="None",
        alpha=0.5,
        marker=".",
        label="Posterior Samples",
    )
    weight = np.ones_like(posterior) / posterior.max()
    axs["histx"].hist(
        posterior,
        bins=30,
        density=True,
        alpha=0.5,
        color="C2",
        weights=weight
    )

    axs["histy"].hist(
        exp.gp.predict(posterior),
        bins=30,
        density=True,
        alpha=0.5,
        color="C2",
        orientation="horizontal",
    )

    ax.set(
        xlabel=r"$\Gamma_\gamma(E=4 keV)$, eV",
        ylabel=r"$k_{\text{eff}}$" if exp.type == "integral" else r"$\chi^2$",
        # ylim=(0,None) if exp.type == 'microscopic' else None,
    )
    ax.legend()

    axs["histx"].set(xlim=ax.get_xlim())
    ax.spines[["right", "top"]].set_visible(False)
    axs["histx"].spines[["left", "bottom", "right", "top"]].set_visible(False)
    axs["histy"].spines[["left", "bottom", "right", "top"]].set_visible(False)

    axs["histy"].set(ylim=ax.get_ylim())

    fig.savefig(save_path, dpi=300)

    return fig


# def output_twinscatterpdf_plot(exp, prior, posterior, save_path):
#     """
#     Plots the prior and posterior predictive distributions with
#     marginal histograms inside the main plot area.
#     """
#     fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

#     ax_histx = ax.twinx()
#     ax_histy = ax.twiny()

#     ax.set_zorder(10)
#     ax.patch.set_visible(False)
#     ax_histx.set_zorder(1)
#     ax_histy.set_zorder(1)

#     if exp.type == "integral":
#         ax.axhline(exp.y_meas, ls="--", label="Measurement", color="C0", zorder=10)
#         ax.axhspan(
#             exp.y_meas - 2 * exp.y_err,
#             exp.y_meas + 2 * exp.y_err,
#             facecolor="C0",
#             alpha=0.2,
#             zorder=9,
#         )

#     prior_y = exp.gp.predict(prior)
#     post_y = exp.gp.predict(posterior)

#     # Scatter Plots
#     ax.plot(
#         prior,
#         prior_y,
#         color="C1",
#         alpha=0.5,
#         ls="None",
#         marker=".",
#         label="Prior Samples",
#         zorder=10,
#     )

#     ax.plot(
#         posterior,
#         post_y,
#         color="C2",
#         ls="None",
#         alpha=0.5,
#         marker=".",
#         label="Posterior Samples",
#         zorder=10,
#     )

#     # ---------------------------------------------------------
#     # PLOT HISTOGRAMS (Twin Axes)
#     # ---------------------------------------------------------

#     # Factor to control "not too high".
#     # 3.0 means the tallest bar will take up 1/3 of the plot height/width
#     height_factor = 4.0

#     # --- Hist X (Bottom) ---
#     ax_histx.hist(prior, bins=30, density=True, alpha=0.3, color="C1")
#     ax_histx.hist(posterior, bins=30, density=True, alpha=0.3, color="C2")

#     # Adjust limits to squash histogram down
#     ymin, ymax = ax_histx.get_ylim()
#     ax_histx.set_ylim(0, ymax * height_factor)
#     ax_histx.axis("off")  # Hide ticks and spines for the histogram axis

#     # --- Hist Y (Left/Right) ---
#     ax_histy.hist(prior_y, bins=30, density=True, alpha=0.3, color="C1", orientation="horizontal")
#     ax_histy.hist(post_y, bins=30, density=True, alpha=0.3, color="C2", orientation="horizontal")

#     # Adjust limits to squash histogram to the side
#     xmin, xmax = ax_histy.get_xlim()
#     ax_histy.set_xlim(0, xmax * height_factor)
#     ax_histy.axis("off")  # Hide ticks and spines

#     # ---------------------------------------------------------
#     # STYLING
#     # ---------------------------------------------------------

#     ax.set(
#         xlabel=r"$\Gamma_\gamma(E=4 keV)$, eV",
#         ylabel=r"$k_{\text{eff}}$" if exp.type == "integral" else r"$\chi^2$",
#     )

#     # Clean spines on the main plot
#     ax.spines[["right", "top"]].set_visible(False)

#     ax.legend(loc="upper right")  # Ensure legend doesn't overlap data if possible

#     fig.suptitle(f"{exp.id}")
#     fig.savefig(save_path, dpi=300)

#     return fig


# def output_pdf_plot(exp, prior, posterior, save_path):
#     """
#     Plots the prior and posterior predictive distributions for a given experiment.
#     """
#     fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
#     ax.hist(
#         np.random.normal(exp.y_meas, exp.y_err, 10000),
#         bins=30,
#         density=True,
#         alpha=0.5,
#         label="Measurement",
#     )

#     ax.hist(exp.gp.predict(prior), bins=30, density=True, alpha=0.5, label="Prior Samples")

#     ax.hist(exp.gp.predict(posterior), bins=30, density=True, alpha=0.5, label="Posterior Samples")

#     ax.set(
#         title=f"{exp.id}",
#         xlabel=r"$k_{\text{eff}}$" if exp.type == "integral" else r"$\chi^2$",
#         ylabel="Density",
#     )

#     ax.legend()
#     fig.savefig(save_path, dpi=300)


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

    # 2. Load Data (Arviz reads the NetCDF we saved earlier)
    idata = az.from_netcdf(netcdf_path)

    # Get parameter names for clearer logging
    param_names = list(idata.posterior.data_vars.keys())
    n_params = len(param_names)
    print(f"Parameters found: {param_names}")

    # 3. Initialize Evaluators for each Experiment
    models = []

    print("\n   Loading Experiment Models...")
    for exp in cfg["experiments"]:
        gp_path = os.path.join("models", f"{exp['id']}_gp.joblib")

        try:
            if exp["type"] == "integral":
                exp_obj = IntegralExperiment(
                    exp_id=exp["id"],
                    exp_type=exp["type"],
                    gp_path=gp_path,
                    y_meas=exp["experimental_data"]["measurement"],
                    y_err=exp["experimental_data"]["uncertainty"],
                )
            elif exp["type"] == "microscopic":
                exp_obj = MicroscopicExperiment(
                    exp_id=exp["id"],
                    exp_type=exp["type"],
                    gp_path=gp_path,
                )
            models.append(exp_obj)
            print(f"    - Loaded {exp['id']}")
        except FileNotFoundError:
            print(f"    !! Skipping {exp['id']}: GP Model not found.")

    # Generate prior samples for predictive checks
    n_prior_samples = 10000
    prior_means = np.array(cfg["parameters"]["prior_means"])
    prior_stds = np.array(cfg["parameters"]["prior_stds"]) * prior_means
    prior_X_samples = np.random.normal(
        loc=prior_means, scale=prior_stds, size=(n_prior_samples, n_params)
    )

    # Extract posterior samples
    burnin = cfg["defaults"]["mcmc"]["burn_in"]

    # select draws after burnin and stack chain/draw into a single sample axis
    posterior_sel = idata.posterior.isel(draw=slice(burnin, None))
    posterior_stacked = posterior_sel.stack(sample=("chain", "draw"))

    # collect parameter columns and build a (nsamples, n_params) array
    param_arrays = [posterior_stacked[param].values.reshape(-1) for param in param_names]
    posterior_X_samples = np.vstack(param_arrays).T

    # ---------------------------------------------------------
    # Plot 1: Trace Plot (Convergence Check)
    # ---------------------------------------------------------
    print("Generating Trace Plot...")
    trace_plot(idata, cfg["project_name"], os.path.join(figures_dir, "trace_plot.png"))

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
    input_pdf_plot(
        prior_X_samples, posterior_X_samples, os.path.join(figures_dir, "input_pdf_plot.png")
    )

    # ---------------------------------------------------------
    # 4. Plot prior and posterior output responses
    # ---------------------------------------------------------
    print("Generating Output Plots...")
    for exp in models:
        # output_scatter_plot(
        #     exp,
        #     prior_X_samples,
        #     posterior_X_samples,
        #     os.path.join(figures_dir, f"{exp.id}_output_scatter_plot.svg"),
        # )

        output_scatterpdf_plot(
            exp,
            prior_X_samples,
            posterior_X_samples,
            os.path.join(figures_dir, f"{exp.id}_output_scatterpdf_plot.png"),
        )

        # output_twinscatterpdf_plot(
        #     exp,
        #     prior_X_samples,
        #     posterior_X_samples,
        #     os.path.join(figures_dir, f"{exp.id}_output_twinscatterpdf_plot.png"),
        # )

        # output_pdf_plot(
        #     exp,
        #     prior_X_samples,
        #     posterior_X_samples,
        #     os.path.join(figures_dir, f"{exp.id}_output_pdf_plot.png"),
        # )

    # ---------------------------------------------------------
    # 4. Plot prior and posterior output responses together
    # ---------------------------------------------------------
    print("Generating Combined Output Plot...")
    fig, axs = plt.subplots(1, len(models), figsize=(5 * len(models), 4), layout="constrained")

    for i, exp in enumerate(models):
        output_scatter_plot(exp, prior_X_samples, posterior_X_samples, None, axs[i])

    fig.savefig(f"{figures_dir}/combined_output_scatter.png", dpi=300)

    # ---------------------------------------------------------
    # 5. Save Text Summary
    # ---------------------------------------------------------
    print("Generating Summary Table...")
    summary_df = az.summary(idata, hdi_prob=0.95)

    summary_path = os.path.join(figures_dir, "posterior_summary.csv")
    summary_df.to_csv(summary_path)

    print("\n--- Prior information ---")
    print(prior_means, prior_stds)

    print("\n--- Posterior information ---")
    print(summary_df["mean"].values[0], (summary_df["sd"] / summary_df["mean"]).values[0])

    print("\n--- Calibration Results ---")
    print(summary_df[["mean", "sd", "r_hat"]])
    print(f"Full summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    plot_mcmc_results(args.config)
