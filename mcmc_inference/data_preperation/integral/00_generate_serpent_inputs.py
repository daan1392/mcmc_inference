import argparse
import pandas as pd
import numpy as np

# Configure parameters
title = "hmi001"
template_file = "data/raw/hmi001/hmi001-template.ser"
nsamples = 100
ace = "/srv/sci/pack/nuclear-data/jeff_40/xsdir/xsdata.jeff40"
ace_perturbed = "/scratch/s1/dhouben/epfl/cr53/aces/xsdata_cr53.jeff40"
output_dir = "data/raw/hmi001/hmi001_serpent_inputs/"

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