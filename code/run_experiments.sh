#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# Train YOLO on Udacity dataset
python train.py -c config/udacity_yolo_baseline.py -e baseline_yolo

# Train Tiny-YOLO on TT100K for detection
python train.py -c config/tt100k_tiny_yolo.py -e baseline_tiny_yolo

# Train Tiny-YOLO on Udacity dataset
python train.py -c config/udacity_tiny_yolo_baseline.py -e baseline_tiny_yolo