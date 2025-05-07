#!/bin/bash

# Default Python version if not specified
PYTHON_VERSION=${1:-"3.12.0"}

# Get current Python version or "0" if not found
CURRENT_VERSION=$(python --version 2>&1 | cut -d' ' -f2 || echo "0")

# Make sure the correct Python version is loaded
if ! command -v python &> /dev/null || [[ "$(python --version 2>&1 | cut -d' ' -f2)" != "$PYTHON_VERSION" ]]; then
    echo "Python $PYTHON_VERSION not found, loading required module..."
    module load stack/2024-06 python/$PYTHON_VERSION
else
    echo "Python $CURRENT_VERSION is already available"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv package manager not found, installing with recommended method"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check connectivity before trying to sync dependencies
check_internet_connectivity() {
    nc -zw1 8.8.8.8 443 2>/dev/null
}

if check_internet_connectivity; then
    echo "Internet connection available, syncing project dependencies..."
    uv sync
else
    echo "No internet connection detected, using existing environment"
    echo "Note: You may need to run 'uv sync' manually when connected"
fi

echo "Setup completed successfully with Python $(python --version 2>&1 | cut -d' ' -f2)!"