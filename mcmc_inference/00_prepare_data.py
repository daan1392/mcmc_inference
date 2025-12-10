import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/ntof/ntof_inputs.csv")
df = df / 1000  # Convert from meV to eV

df.to_csv("data/processed/ntof/ntof_inputs_eV.csv", index=False)
