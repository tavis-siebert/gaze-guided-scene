#!/bin/bash

python3 -m venv --system-site-packages venv
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

echo "Setup completed successfully with Python $(python3 --version 2>&1 | cut -d' ' -f2)!"