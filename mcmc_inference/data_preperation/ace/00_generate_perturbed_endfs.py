import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Configure parameters
output_folder = "data/raw/cr53_ace/tapes/"
unperturbed_file = "data/raw/cr53_ace/n_24-Cr-053g_GG1.jeff"
nsmp = 5
placeholder = "{GG1}"

def to_endf6_number(number):
    """
    Converts a float number to the format 'mantissa+exponent'
    with 6 decimals in the mantissa and a positive or negative integer exponent.
    
    Example:
        956.6001 -> '9.566001+2'
        0.01234  -> '1.234000-2'
    """
    import math

    if number == 0:
        return "0.000000+0"
    
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10 ** exponent)
    if number < 1:
        return f"{mantissa:.6f}{exponent}"
    else:
        return f"{mantissa:.6f}+{exponent}"

samples = np.linspace(1e-5, 20.0, nsmp)

# Read file once
with open(unperturbed_file, "r", encoding="utf-8") as f:
    text = f.read()

# Save the parameters to a csv file
df_samples = pd.Series(samples, name="GG1_samples")
df_samples.rename("GG1", inplace=True)
df_samples
df_samples.to_csv(f"{output_folder}/GG1_samples.csv", index=False)

# Create perturbed ENDF-6 files for each sampled parameter
for i, GG1 in enumerate(samples):
    perturbed_endf6 = text.replace(placeholder, to_endf6_number(GG1))
    print(to_endf6_number(GG1))
    with open(f"{output_folder}/n_24-Cr-053g.jeff{i}", "w", encoding="utf-8") as f_out:
        f_out.write(perturbed_endf6)