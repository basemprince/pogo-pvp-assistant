#!/bin/bash

ENV_NAME="pogo-pvp-assistant"

# Detect conda executable path
CONDA_PATH=$(which conda)

if [ -z "$CONDA_PATH" ]; then
    echo "Conda not found in PATH. Make sure Anaconda/Miniconda is installed and available."
    exit 1
fi

# Source conda shell hook (portable way)
eval "$("$CONDA_PATH" shell.bash hook 2>/dev/null)" || {
    echo "Error: Could not initialize conda shell. Try running 'conda init' manually."
    exit 1
}

# Deactivate current environment if active
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "Deactivating current environment: $ENV_NAME"
    conda deactivate || echo "Could not deactivate environment"
fi

# Remove env if it exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists. Removing it..."
    if ! conda remove -n "$ENV_NAME" --all -y; then
        echo "Failed to remove environment. Make sure it's not active."
        exit 1
    fi
fi

echo "Creating conda environment..."
conda env create -f config/environment.yml

echo "Activating environment..."
conda activate "$ENV_NAME"

echo "Installing ADB..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install android-platform-tools
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update && sudo apt install -y android-tools-adb
else
    echo "ADB install skipped (unsupported OS: $OSTYPE)"
fi

echo "✅ Setup complete!"
echo "👉 To start using the environment, run: conda activate $ENV_NAME"
