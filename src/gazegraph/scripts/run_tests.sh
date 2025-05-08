#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/tests.out
#SBATCH --time=2:00:00
#SBATCH --gpus=1

# Run all tests, including integration tests that need GPU
uv run pytest -v -rs tests/