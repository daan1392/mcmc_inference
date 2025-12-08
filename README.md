# MCMC Inference

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="CCDS template" />
</a>

Markov Chain Monte Carlo (MCMC) workflows to infer nuclear data from microscopic and integral measurements. Surrogate models are built per measurement using Gaussian Processes.

## Features
- Train Gaussian Process surrogates for measurements
- Run MCMC inference using surrogate predictions
- Config-driven experiments

## Prerequisites
- Python 3.10+
- Git
- Recommended: virtual environment (venv or conda)

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
1. Place measurement and surrogate training data in data/ (e.g. data/raw or data/inputs).
2. Edit a config in configs/config.yaml (set experiment path, data paths, hyperparameters).
3. Run the main script:
```bash
python scripts/run_complete.py --config configs/config.yaml
```
Or set environment variables / CLI args as supported by the script.

## Tips
- Keep each experiment in its own output directory to avoid overwriting results.
- Use "RBF" kernel for GP to approximate semi-linear functions and the "Quadratic" kernel for quadratic functions.
- Always very 

## Contributing / Support
Open issues or PRs on the repository for bugs, feature requests, or questions.
