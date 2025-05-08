#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/convert.out
#SBATCH --time=24:00:00

uv run scripts/convert_graphs.py