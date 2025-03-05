#!/bin/bash

# Default Python version if not specified
PYTHON_VERSION=${1:-"3.11.6"}

# Get current Python version or "0" if not found
CURRENT_VERSION=$(python --version 2>&1 | cut -d' ' -f2 || echo "0")

# Make sure the correct Python version is loaded
if ! command -v python &> /dev/null || [[ ! "$(echo -e "$CURRENT_VERSION\n$PYTHON_VERSION" | sort -V | tail -n1)" = "$CURRENT_VERSION" ]]; then
    echo "Python $PYTHON_VERSION not found, loading required module..."
    module load stack/2024-06 python/$PYTHON_VERSION
else
    echo "Python $CURRENT_VERSION is already available"
fi

python -m venv --system-site-packages venv
source venv/bin/activate

check_internet_connectivity() {
    nc -zw1 8.8.8.8 443 2>/dev/null
}

if check_internet_connectivity; then
    echo "Internet connection available, installing/upgrading packages..."
    pip3 install -r requirements.txt --upgrade
else
    echo "No internet connection detected, skipping package installation"
    echo "Note: You may need to run 'pip3 install -r requirements.txt --upgrade' manually when connected"
fi

echo "Setup completed successfully with Python $(python --version 2>&1 | cut -d' ' -f2)!"