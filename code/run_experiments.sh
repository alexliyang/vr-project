#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# DeepLab

python train.py -c config/camvid_deeplabv2_bilinear_wd.py -e deeplabv2_bilinear_wd


# Tiramisu

python train.py -c config/camvid_tiramisu_fc67_enhanced.py -e tiramisu_fc67_enhanced

python train.py -c config/camvid_tiramisu_fc67_enhanced_finetune.py -e tiramisu_fc67_enhanced_finetune

python train.py -c config/camvid_tiramisu_fc56_enhanced.py -e tiramisu_fc56_enhanced

python train.py -c config/camvid_tiramisu_fc56_enhanced_finetune.py -e tiramisu_fc56_enhanced_finetune
