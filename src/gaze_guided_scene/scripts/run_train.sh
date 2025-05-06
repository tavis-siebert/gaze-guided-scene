#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/train.out
#SBATCH --time=12:00:00

uv run src/gaze_guided_scene/main.py train --device gpu --task future_actions