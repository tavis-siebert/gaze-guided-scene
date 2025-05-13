#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/build.out
#SBATCH --time=48:00:00

./scripts/run.sh build-graphs --device gpu --enable-tracing --videos OP01-R04-ContinentalBreakfast --overwrite