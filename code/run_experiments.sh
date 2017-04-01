#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# FCN

# Train on CamVid
#python train.py -c config/camvid_fcn.py -e fcn_baseline

# Train on CityScapes
#python train.py -c config/cityscapes_fcn.py -e fcn_baseline


# SEGNET

# Train on CamVid
python train.py -c config/camvid_segnet.py -e segnet_baseline_scratch
