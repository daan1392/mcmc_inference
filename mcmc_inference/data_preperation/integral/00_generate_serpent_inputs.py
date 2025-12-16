import argparse
import pandas as pd
import numpy as np

# Configure parameters
title = "pmi002"
template_file = "data/raw/pmi002/pmi002-template.ser"
nsamples = 5
ace = "/home/houben/nuclear-data/jeff40/jeff4.xsdir.serpent"
ace_perturbed = "/home/houben/nuclear-data/cr53/aces_10-24/xsdata_cr53.jeff40"
output_dir = "data/raw/pmi002/pmi002_serpent_inputs/"

# Read template file
with open(template_file, "r", encoding="utf-8") as f:
        template = f.read()

template = template.replace("{ace}", ace)
template = template.replace("{perturbed_ace}", ace_perturbed)

# Write files 
for i in range(nsamples):
    output_path = f"{output_dir}{title}.ser{i}"
    with open(output_path, "w", encoding="utf-8") as f_out:
        # replace placeholder with sample index
        text = template.replace("{i}", str(i))
        f_out.write(text)