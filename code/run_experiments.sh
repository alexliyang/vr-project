#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# FCN

# Train on CamVid
#python train.py -c config/camvid_fcn.py -e fcn_baseline

# Train on CityScapes
#python train.py -c config/cityscapes_fcn.py -e fcn_baseline

#Train DeepLabv2 in CamVid using adam optimizer
python train.py -c config/camvid_deeplabv2_adam.py -e deeplabv2_adam

#Train DeepLabv2 in CamVid using adam optimizer and preprocessing
python train.py -c config/camvid_deeplabv2_adam_preprocessing.py -e deeplabv2_adam_preprocessing