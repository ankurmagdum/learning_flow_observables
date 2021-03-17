#!bin/bash

bsub -n 128 -W 4:00 -R "rusage[mem=10000]" -r mpirun python tune.py
