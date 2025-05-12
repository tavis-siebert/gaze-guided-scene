#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/train.out
#SBATCH --time=48:00:00

./run.sh train --device gpu --task future_actions --object-node-feature roi-embeddings --load-cached