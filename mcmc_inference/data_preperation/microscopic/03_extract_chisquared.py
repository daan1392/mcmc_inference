import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# plt.rcParams.update(
#     {
#         "text.usetex": False,
#         "font.family": "STIXGeneral",
#         "mathtext.fontset": "cm",
#         "axes.formatter.use_mathtext": True,
#         "font.size": 16,
#     }
# )

def extract_i_chi_squared(file_path, position, pat="CHI SQUARED DIVIDED BY NDAT"):
    """Extract the I-th occurrence of the CHI_SQUARED line from the given file."""
    pattern = re.compile(pat)

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        count = 0
        for _, line in enumerate(fh, 1):
            if pattern.search(line):
                count += 1
                if count == position:
                    return float(line.split()[-1])
    return None

def extract_chi_squared(title, input_folder, results_folder, N):
    """
    Extract chi-squared values from a set of files, 
    plot them and save it to a csv file.
    """
    lpt_pattern = results_folder + title + "{}.lpt"

    chi_squared = []
    chi_squared_reduced = []
    for i in tqdm(range(N), desc="Extracting chi-squared"):
        lpt_file = lpt_pattern.format(i)
        chi_squared.append(extract_i_chi_squared(lpt_file, 1, pat="CHI SQUARED ="))
        chi_squared_reduced.append(extract_i_chi_squared(lpt_file, 1, pat="CHI SQUARED DIVIDED BY NDAT ="))


    chi_squared = pd.Series(chi_squared, name='chi_squared')
    chi_squared.to_csv(f"{results_folder}chi_squared_values.csv", index=False)

    chi_squared_reduced = pd.Series(chi_squared_reduced, name='reduced_chi_squared')
    chi_squared_reduced.to_csv(f"{results_folder}reduced_chi_squared_values.csv", index=False)

    parameters = pd.read_csv(f"{input_folder}perturbed_parameters/perturbed_resonances.csv")

    # Plot chi-squared vs perturbed parameter
    fig, ax = plt.subplots(figsize=(5,4))

    ax.plot(parameters, chi_squared, marker='x', linestyle='', label='nTOF measurements')

    ax.set(
        xlabel=r'$\Gamma_\gamma$ (E=4033 eV), eV',
        ylabel=r'$\chi^2$',
        title='Chi-squared vs Perturbed Parameters',
    )

    ax.spines[["right", "top"]].set_visible(False)

    ax.legend()

    fig.savefig(f"{results_folder}chi_squared_vs_perturbed_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    title = "cr53_thin"
    input_folder = "data/raw/Pérez-Maroto(2025)/cr53_thin_training/"
    results_folder = "data/raw/Pérez-Maroto(2025)/cr53_thin_training/results/"
    N = 10

    extract_chi_squared(title, input_folder, results_folder, N)