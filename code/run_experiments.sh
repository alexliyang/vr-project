#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# Tiramisu

python train.py -c config/camvid_tiramisu_fc103_enhanced.py -e tiramisu_fc103_enhanced

python train.py -c config/camvid_tiramisu_fc103_enhanced_finetune.py -e tiramisu_fc103_enhanced_finetune

# DeepLab

python train.py -c config/camvid_deeplabv2_bilinear_wd -e deeplabv2_bilinear_wd