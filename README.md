# MCMC Inference
Markov Chain Monte Carlo (MCMC) framework to infer nuclear data from both microscopic energy dependent and integral experiments. Surrogate models are built per experiment using Gaussian Processes with the 80/20 rule for testing. This work was build upon scripts from @Sarah Maccario. Microscopic experiments could be included due to the help of Pablo Pérez-Maroto. 

## Features
- Train Gaussian Process (GP) surrogates for each experiment included
- Run MCMC inference using the GPs
- Plot prior and posterior responses

## Installation
Clone and install in editable mode:
```bash
git clone https://github.com/daan1392/mcmc_inference.git
cd mcmc_inference
pip install -e .
```

## Project layout (key folders)
- data/           — raw and processed measurement & training data
- configs/        — experiment/configuration YAML files
- scripts/        — helper/run scripts
- src/            — package source (models, inference, utils)

## Usage
1. Place measurement and surrogate training data in data/ (e.g. data/processed).
2. Edit a config in configs/config.yaml (set experiment path, data paths, measurement data etc.).
3. Run the main script:
```bash
python scripts/run_complete.py --config configs/config.yaml
```

## Tips
- Keep each experiment in its own output directory to avoid overwriting results.
- Use "RBF" kernel to approximate quasi-linear functions and the "Quadratic" kernel for quadratic functions.
- Use sufficient training points and observe the validation plots and printed summary to assess the quality of the GPs