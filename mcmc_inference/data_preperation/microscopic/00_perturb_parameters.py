import numpy as np
import pandas as pd

def to_sammy_number(number):
    """
    Convert a number to SAMMY format:
    d.dddddd±e  (10 characters total)
    """
    s = f"{number:.6E}" 
    mantissa, exponent = s.split("E")

    exp = int(exponent)

    if not -9 <= exp <= 9:
        raise ValueError("Currently only single digit exponents are supported.")
    
    return f"{mantissa}{exp:+d}"

def perturb_parameter(par_file, output_folder, N):
    samples = pd.Series(sampler(sample_range[0], sample_range[1], N)).rename("Gg0")
    (samples/1000).to_csv(output_folder + "perturbed_resonances.csv", index=False) # SAMMY uses meV in .par files

    # Read template parameter file once
    with open(par_file, "r", encoding="utf-8") as f:
        text = f.read()

    output_pattern = output_folder + "cr53_thin_{}.par"

    # Create perturbed ENDF-6 files for each sampled parameter
    for i, Gg0 in enumerate(samples):
        perturbed_endf6 = text.replace(placeholder, to_sammy_number(Gg0))
        with open(output_pattern.format(i), "w", encoding="utf-8") as f_out:
            f_out.write(perturbed_endf6)

if __name__ == "__main__":
    par_file = r"data/raw/Pérez-Maroto(2025)/cr53_thin_training/cr53_thin_jeff40_p.par"
    samples = 10
    output_folder = r"data/raw/Pérez-Maroto(2025)/cr53_thin_training/perturbed_parameters/"

    # Configure settings
    placeholder = "{placehol}"
    sample_range = (0, 9999)
    sampler = np.random.uniform

    print(f"Generating {samples} perturbed parameter files in {output_folder}...")
    perturb_parameter(par_file, output_folder, samples)