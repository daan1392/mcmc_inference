import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import serpentTools as st
import argparse

def analyze_serpent_results(inputs, title, output_csv, N):
    all_columns = []
    all_columns.extend([f'{title}', f'{title}_std'])

    outputs = pd.DataFrame(index=range(num_samples), columns=all_columns)

    print(f"--- Processing simulation set: {title} ---")

    for i in range(num_samples):
        print(f'Processing sample {i+1}/{num_samples}', end='\r')
        
        file_path = f'data/raw/pmi002/results/{title}.ser{i}_res.m'

        try:
            sim = st.read(file_path)
            
            k_eff, k_eff_std = sim.resdata['impKeff']

            target_cols = [f'{title}', f'{title}_std']
            outputs.loc[i, target_cols] = [k_eff, k_eff_std]
        except Exception as e:
            print(f"\nError processing file '{file_path}': {e}")
            outputs.loc[i, [f'{title}', f'{title}_std']] = [np.nan, np.nan]

    print(f'\nFinished processing all samples for {title}.\n')

    print(f"Saving results to '{output_csv_file}'...")
    outputs.to_csv(output_csv_file, index=False)

if __name__ == "__main__":
    # Configure parameters
    inputs = pd.read_csv(r"data/raw/cr53_ace/tapes/GG1_samples.csv", index_col=0).to_numpy().flatten()
    title = "pmi002"
    num_samples = 69
    output_csv_file = f"data/raw/pmi002/{title}_outputs.csv"

    analyze_serpent_results(inputs, title, output_csv_file, num_samples)