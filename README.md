# MCMC Inference
Markov Chain Monte Carlo (MCMC) framework to infer nuclear data from both microscopic energy dependent and integral experiments. Surrogate models are built per experiment using Gaussian Processes with the 80/20 rule for testing. 

## Acknowledgements
This work was build upon scripts from a PhD student at EPFL, Sarah Maccario. Thanks to Pablo Pérez-Maroto, the capture yield measurement of Cr-53 could be included. He provided the required input files for running SAMMY. 

## Features
- Generate training data (perturbed ACE files, )
- Train Gaussian Process (GP) surrogates for each experiment included
- Run MCMC inference using the GPs
- Plot prior and posterior responses

## Installation
Clone and install:
```bash
git clone https://github.com/daan1392/mcmc_inference.git
cd mcmc_inference
pip install .
```

## Project layout
- data/           (experiment data, training data)
- configs/        (configuration file to specify inference settings)
- scripts/        (scripts to run specific workflows)
- mcmc_inference/ (python functions and code)
- reports/        (figures and written reports)

## Usage
1. Place measurement and surrogate training data in data/ (e.g. data/processed).
2. Edit a config in configs/config.yaml (set experiment path, data paths, measurement data etc.).
3. Run the main script:
```bash
python scripts/run_inference.py --config configs/config.yaml
```

## Tips
- Keep each experiment in its own output directory to avoid overwriting results.
- Use "RBF" kernel to approximate quasi-linear functions and the "Quadratic" kernel for quadratic functions.
- Use sufficient training points and observe the validation plots and printed summary to assess the quality of the GPs