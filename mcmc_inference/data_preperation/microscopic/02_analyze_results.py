import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams.update(
#     {
#         "text.usetex": False,
#         "font.family": "STIXGeneral",
#         "mathtext.fontset": "cm",
#         "axes.formatter.use_mathtext": True,
#         "font.size": 16,
#     }
# )

if __name__ == "__main__":
    title = "cr53_thin"
    results_folder = "data/raw/Pérez-Maroto(2025)/cr53_thin_training/results/"
    N = 50
    plot_pattern = results_folder + "cr53_thin{}.lst"

    data = pd.read_csv(plot_pattern.format(0), sep='\s+', header=None)
    data.columns = ['e', 'x', 'dx', 'prior_th']

    data.set_index('e', inplace=True)

    data_df = pd.DataFrame()
    for i in range(N):
        data = pd.read_csv(plot_pattern.format(i), sep='\s+', header=None)
        data.columns = ['e', 'x', 'dx', 'prior_th']
        data.set_index('e', inplace=True)
        data_par = data['prior_th']
        data_par.rename(i, inplace=True)
        data_df = pd.concat([data_df, data_par], axis=1)

    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

    ax.errorbar(data.index, data['x'], yerr=data['dx'], 
                label='2025 Pérez-Maroto', fmt='o', capsize=5, elinewidth=2, color='black',zorder=0)

    for i in range(N):
        ax.plot(data_df.index, data_df[i], marker='', linestyle='--', lw=0.5)

    ax.set(
        title=title,
        xlabel='Incident neutron energy, eV',
        ylabel=r'Capture yield',
        xlim=(1e3, 1e4),
        # xscale='log',
        # yscale='log'
    )
    # ax.spines[["right", "top"]].set_visible(False)

    ax.legend(loc='upper left')

    fig.savefig(f'{results_folder}{title}.png')
    plt.close()