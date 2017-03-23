#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# Train SSD300 on TT100K detection dataset
python train.py -c config/tt100k_ssd300.py -e baseline_ssd300

# Train SSD300 on Udacity dataset
python train.py -c config/udacity_ssd300.py -e baseline_ssd300