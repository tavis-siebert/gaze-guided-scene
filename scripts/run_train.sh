#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/train.out
#SBATCH --time=48:00:00

./scripts/run.sh train --task action_recognition