import sandy
from tqdm import tqdm
import argparse
import endf
import pandas as pd
import matplotlib.pyplot as plt
import re

# Configure parameters
title = "cr53_aces_capture"
output_folder = "data/raw/cr53_ace/aces/"
samples = 5

# Create function to read ace files into a dataframe
def read_aces(folder_path, N, zai, mt, T=0):
    """
    Given the folder where ace files are stored, 
    return a dataframe with the pointwise cross sections.
    """
    xs_list = pd.DataFrame()

    T = str(T)+"K"
    for i in tqdm(range(N), ascii=True, desc='Reading ACE files'):
        path = f"{folder_path}{zai}.03{i}c"
        ace = endf.incident_neutron.IncidentNeutron.from_ace(path)
        xs = pd.Series(ace[mt].xs[T].y, index=ace[mt].xs[T].x, name=i)
        xs_list = pd.concat([xs_list, xs], join='outer', axis=1)
    xs_list.index.name = 'Energy (eV)'
    xs_list.columns.name = 'Sample'

    return xs_list.sort_index(axis=0).interpolate('slinear')

for i in tqdm(range(samples), ascii=True, desc="Generating ACE files"):
    tape = sandy.Endf6.from_file(f"tapes/n_24-Cr-053g.jeff{i}")
    ace = tape.get_ace(temperature=300, err=0.01, minimal_processing=True)
    file2write=open(f"aces/240530.03{i}c",'w')
    file2write.write(ace["ace"])
    file2write.close()

file2write=open(f"{output_folder}xsdata_cr53.jeff40",'w')
for i in tqdm(range(samples), ascii=True, desc="Generating xsdata entry"):
    file2write.write(f"24053.03{i}c  24053.03{i}c  1  24053  0  52.940658  300  0  /home/houben/cr53/aces/240530.03{i}c\n")
file2write.close()

aces = read_aces(output_folder, samples, zai=240530, mt=102, T=300)

fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

ax.plot(aces, lw=0.5, marker='o', ms=0.5)

ax.set(
    xlim=(1e3, 10e3),
    ylim=(0, 2.5),
    xlabel='Energy, eV',
    ylabel='Cross section, b',
)

ax.spines[["right", "top"]].set_visible(False)

fig.savefig(f"{output_folder}{title}.png", dpi=300)
plt.close()