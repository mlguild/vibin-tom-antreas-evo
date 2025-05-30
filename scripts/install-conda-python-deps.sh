#!/bin/bash

# 🐍 Archivum Conda Environment Setup Script
# This script sets up a dedicated Conda environment for the Archivum project

set -e  # Exit on any error

CONDA_ENV_NAME="archivum"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Setting up Conda environment for Archivum..."
echo "📁 Project root: $PROJECT_ROOT"

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "✅ Conda found: $(conda --version)"
        return 0
    else
        echo "❌ Conda not found"
        return 1
    fi
}

# Function to install Miniconda
install_miniconda() {
    echo "📦 Installing Miniconda..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if [[ $(uname -m) == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    else
        echo "❌ Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    # Download and install Miniconda
    TEMP_DIR=$(mktemp -d)
    INSTALLER_PATH="$TEMP_DIR/miniconda.sh"
    
    echo "⬇️  Downloading Miniconda from $MINICONDA_URL"
    curl -L -o "$INSTALLER_PATH" "$MINICONDA_URL"
    
    echo "🔧 Installing Miniconda..."
    bash "$INSTALLER_PATH" -b -p "$HOME/miniconda3"
    
    # Initialize conda
    echo "🔄 Initializing Conda..."
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null || true
    
    # Add conda to PATH for this session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    echo "✅ Miniconda installed successfully!"
    echo "⚠️  Please restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
}

# Function to create conda environment
create_conda_env() {
    echo "🏗️  Creating Conda environment: $CONDA_ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "⚠️  Environment '$CONDA_ENV_NAME' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            echo "📦 Using existing environment..."
            return 0
        fi
    fi
    
    # Create environment with Python 3.11
    echo "🐍 Creating new environment with Python 3.11..."
    conda create -n "$CONDA_ENV_NAME" python=3.11 -y
    
    echo "✅ Environment '$CONDA_ENV_NAME' created successfully!"
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing project dependencies..."
    
    # Activate environment
    echo "🔄 Activating environment: $CONDA_ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    # Install uv for fast package management
    echo "⚡ Installing uv for fast package management..."
    pip install uv
    
    # Install project dependencies using uv
    echo "📚 Installing project dependencies from pyproject.toml..."
    cd "$PROJECT_ROOT"
    uv pip install .
    
    echo "✅ Dependencies installed successfully!"
}

# Function to save environment configuration
save_env_config() {
    echo "💾 Saving environment configuration..."
    
    CONFIG_FILE="$PROJECT_ROOT/00_meta/archivum_env_config.yaml"
    cat > "$CONFIG_FILE" << EOF
# Archivum Environment Configuration
# Generated on $(date)

environment:
  type: conda
  name: $CONDA_ENV_NAME
  python_version: "3.11"
  created_date: "$(date +%Y-%m-%d)"
  
activation_command: |
  conda activate $CONDA_ENV_NAME
  
notes: |
  This environment was created using the install-conda-python-deps.sh script.
  To activate: conda activate $CONDA_ENV_NAME
  To deactivate: conda deactivate
EOF
    
    echo "✅ Configuration saved to: $CONFIG_FILE"
}

# Main execution
main() {
    echo "🎯 Starting Archivum Conda setup..."
    
    # Check if conda is available, install if not
    if ! check_conda; then
        echo "🔧 Conda not found. Installing Miniconda..."
        install_miniconda
        
        # Re-check conda availability
        if ! check_conda; then
            echo "❌ Failed to install or find Conda. Please install manually."
            exit 1
        fi
    fi
    
    # Create conda environment
    create_conda_env
    
    # Install dependencies
    install_dependencies
    
    # Save configuration
    save_env_config
    
    echo ""
    echo "🎉 Archivum Conda environment setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "   2. Activate the environment: conda activate $CONDA_ENV_NAME"
    echo "   3. Verify installation: python -c 'import fire, rich, yaml; print(\"✅ All dependencies working!\")'"
    echo ""
    echo "🔧 Environment details:"
    echo "   Name: $CONDA_ENV_NAME"
    echo "   Python: 3.11"
    echo "   Location: $(conda info --base)/envs/$CONDA_ENV_NAME"
    echo ""
}

# Run main function
main "$@" 