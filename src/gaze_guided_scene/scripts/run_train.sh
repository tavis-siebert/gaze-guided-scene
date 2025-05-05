#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/train.out
#SBATCH --time=12:00:00

uv run main.py train --device gpu --task future_actions