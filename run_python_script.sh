#!/bin/bash

set -e
PYTHONPATH=../../:../../iterative_surrogate_optimization:${PYTHONPATH} python3 train.py "$@" 
										  	
