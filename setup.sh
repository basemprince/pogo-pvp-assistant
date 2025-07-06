#!/bin/bash

set -e

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pogo-pvp-assistant

echo "Installing ADB..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install android-platform-tools
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update && sudo apt install -y android-tools-adb
fi

echo "Done!"
