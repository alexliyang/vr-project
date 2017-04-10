#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.


python train.py -c config/camvid_tiramisu_fc56_enhanced_finetune.py -e tiramisu_fc56_enhanced_finetune

python train.py -c config/camvid_deeplabv2_paper.py -e deeplabv2_paper

python train.py -c config/camvid_segnetbasic_scratch_paper.py -e segnetbasic_scratch_paper