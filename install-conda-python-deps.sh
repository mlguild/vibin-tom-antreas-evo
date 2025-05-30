#!/bin/bash
# install-conda-python-deps.sh - For Micro Model Inference Package

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONDA_DIR="$HOME/miniconda"
ENV_NAME="time-inference"
PYTHON_VERSION="3.11"
MINICONDA_INSTALL_SCRIPT="miniconda.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# --- Miniconda Installation (Only if not already installed) ---
if [ ! -d "$CONDA_DIR/bin" ] || ! command -v conda &> /dev/null; then
    echo "⬇️ Downloading Miniconda..."
    wget --quiet "$MINICONDA_URL" -O "$MINICONDA_INSTALL_SCRIPT"

    echo "📦 Installing Miniconda to $CONDA_DIR..."
    /bin/bash "$MINICONDA_INSTALL_SCRIPT" -b -p "$CONDA_DIR"

    echo "🧹 Cleaning up Miniconda installer..."
    rm "$MINICONDA_INSTALL_SCRIPT"
else
    echo "✅ Miniconda already installed."
fi

# --- Add Conda to PATH for this script's execution ---
export PATH="$CONDA_DIR/bin:$PATH"
echo " PATH updated for script execution."

# --- Conda Initialization for Shells (Run only once usually) ---
echo "⚙️ Attempting Conda initialization for shells (might require sourcing config)..."
conda init bash || echo "Conda init bash failed, might already be initialized or need manual sourcing."
if command -v fish &> /dev/null; then
    conda init fish || echo "Conda init fish failed, might already be initialized or need manual sourcing."
fi

# --- Conda Environment Creation & UV Installation ---
echo "🐍 Ensuring conda environment '$ENV_NAME' exists with Python $PYTHON_VERSION..."
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
else
    echo " Environment '$ENV_NAME' already exists."
    # Optionally, update the environment here if needed
    # conda update -n "$ENV_NAME" --all -y
fi

echo "✨ Installing/Updating uv package manager into '$ENV_NAME'..."
# Use conda run to execute commands within the new environment
# Using update ensures it's present and latest compatible version
conda run -n "$ENV_NAME" conda install -c conda-forge uv -y

# --- Project Package Installation ---
echo "🚀 Installing the micro-model-inference package using uv..."
# Ensure pyproject.toml is in the current directory where this script is run
if [ ! -f pyproject.toml ]; then
    echo "❌ Error: pyproject.toml not found in the current directory."
    echo "   Please run this script from the 'micro-model-inference' directory."
    exit 1
fi
# Install the package defined in pyproject.toml
# Use -e for editable install during development if desired
# conda run -n "$ENV_NAME" uv pip install -e .
conda run -n "$ENV_NAME" uv pip install .

echo "✅ Installation complete."
echo "   To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "   If the 'conda activate' command is not found, you may need to source your shell configuration file first:"
echo "     - For bash: source ~/.bashrc"
echo "     - For fish: source ~/.config/fish/config.fish"
echo "   Then retry 'conda activate $ENV_NAME'." 