#!/bin/bash
module load stack/2024-06 python/3.11.6
python -m venv --system-site-packages venv
source venv/bin/activate
pip3 install -r requirements.txt --upgrade

echo "Setup completed successfully!"