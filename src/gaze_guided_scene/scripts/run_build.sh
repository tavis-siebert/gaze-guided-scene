#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=logs/build.out
#SBATCH --time=24:00:00

uv run src/gaze_guided_scene/main.py build-graphs --device gpu --videos P02-R04-ContinentalBreakfast OP03-R04-ContinentalBreakfast P19-R04-ContinentalBreakfast OP02-R01-PastaSalad P02-R01-PastaSalad P01-R01-PastaSalad P04-R01-PastaSalad P01-R02-TurkeySandwich P06-R02-TurkeySandwich P10-R02-TurkeySandwich OP05-R03-BaconAndEggs P21-R03-BaconAndEggs P20-R03-BaconAndEggs OP02-R03-BaconAndEggs OP04-R07-Pizza OP01-R07-Pizza OP04-R06-GreekSalad OP03-R06-GreekSalad P10-R05-Cheeseburger OP01-R05-Cheeseburger --enable-tracing