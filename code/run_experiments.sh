#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# Train VGG on KITTI dataset
# 1. From scratch
python train.py -c config/kitti_baseline_vgg.py -e baseline_vgg
# 2. Fine-tune on ImageNet weights
python train.py -c config/kitti_finetune_vgg.py -e finetune_vgg

# Train ResNet on KITTI dataset
# 1. From scratch
python train.py -c config/kitti_resnet_baseline.py -e baseline_resnet
# 2. Fine-tune on ImageNet weights
python train.py -c config/kitti_resnet_finetune_imagenet.py -e finetune_resnet

# Train YOLO on TT100K for detection
python train.py -c config/tt100k_detection.py -e baseline_yolo

# Meta-parameter tuning for ResNet on TT100K
python optimization.py