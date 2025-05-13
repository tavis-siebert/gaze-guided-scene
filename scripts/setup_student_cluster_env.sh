#!/bin/bash

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

echo "Setup completed successfully with Python $(python3 --version 2>&1 | cut -d' ' -f2)!"