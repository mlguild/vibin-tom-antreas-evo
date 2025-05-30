# Project Setup

This document outlines the structure and setup for the "Tom Antreas Evolution Vibe Coding" project.

## 1. Overview

The primary goal of this project is to explore various forms of evolution-based optimization methods, contrasting and potentially combining them with gradient-based approaches. The initial phase involves setting up a robust deep learning pipeline for a benchmark task (EMNIST classification) which can then be adapted for evolutionary algorithms.

## 2. Directory Structure

```
tom_antreas_evolution_vibe_coding/
├── .cursorrules        # Configuration and directives for the Chamber AI assistant
├── .gitignore          # Standard Python gitignore
├── docs/               # Project documentation (this folder)
│   ├── README.md
│   ├── project_setup.md
│   ├── dataset_preparation.md
│   ├── model_architecture.md
│   ├── training_process.md
│   └── evolutionary_strategies.md
├── emnist_data/        # (Created dynamically) Stores downloaded EMNIST datasets
│   └── byclass/
│   └── letters/
│   └── ... (other splits)
├── install-conda-python-deps.sh # Script for environment setup
├── papers/             # Summaries and notes on relevant research papers
│   ├── openai_es_starter.md
│   ├── howuhh_es_openai.md
│   └── salimans_et_al_2017_es.md
├── pyproject.toml      # Project metadata, dependencies (using setuptools and pip)
├── README.md           # Main project README
├── run_gradient_based.py # Script to run gradient-based training
├── saved_models/       # (Created dynamically) Stores trained model checkpoints
└── tinkering/          # Python package for experimental code, utilities, and core components
    ├── __init__.py
    ├── emnist_dataset.py # EMNIST dataset loading and augmentation
    ├── models.py         # Model definitions (e.g., ResNet, functional utilities)
    └── train.py          # Training and validation loop functions
```

## 3. Dependencies and Environment

Key dependencies are managed via `pyproject.toml` and can be installed using `pip` in a virtual environment. Refer to `pyproject.toml` for the full list, which includes:

*   `torch` & `torchvision`: For deep learning.
*   `fire`: For creating command-line interfaces.
*   `rich`: For enhanced terminal output (logging, tracebacks, progress bars).
*   `tqdm`: For progress bars (used within `rich.progress`).
*   `pathlib`: For path manipulations.

An `install-conda-python-deps.sh` script is also provided, potentially for an alternative Conda-based setup, though the primary dependency management demonstrated so far uses `pip` with `pyproject.toml`.

## 4. AI Assistant Directives (`.cursorrules`)

The `.cursorrules` file provides specific instructions and context to the AI assistant (Chamber) to guide its behavior and coding style throughout the project. This includes preferences for certain libraries and coding practices.

## 5. Initial Setup Steps (Conceptual)

1.  Clone the repository (if applicable).
2.  Create and activate a Python virtual environment (e.g., using `venv` or `conda`).
3.  Install dependencies: `pip install .` or `pip install -e .` for an editable install.
4.  Review the `install-conda-python-deps.sh` if a Conda environment is preferred. 