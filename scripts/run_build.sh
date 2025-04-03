#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/build.out
#SBATCH --time=12:00:00

source scripts/setup_student_cluster_env.sh

python main.py build --device gpu --videos OP03-R02-TurkeySandwich OP01-R04-ContinentalBreakfast OP03-R01-PastaSalad  --enable-tracing