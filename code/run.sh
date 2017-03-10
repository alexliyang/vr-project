#!/bin/bash

# Train VGG on TT100K dataset from scratch
# 1. Baseline configuration
python train.py -c config/tt100k_classif.py -e baseline_vgg
# 2. Resize to 256, 256 and then take random crops of 224, 224
python train.py -c config/tt100k_classif_crop.py -e crop_vgg
# 3. Substract mean and divide by std computed on the train set as image preprocessing
python train.py -c config/tt100k_classif_preprocess.py -e preprocess_vgg
# 4. Transfer learning
# TODO

# Train VGG on KITTI dataset
# 1. From scratch
# TODO
# 2. Fine-tune on ImageNet weights
# TODO

# Train ResNet on TT100K dataset
# 1. From scratch
python train.py -c config/tt100k_resnet_baseline.py -e baseline_resnet
# 2. Fine-tune on ImageNet weights
python train.py -c config/tt100k_resnet_baseline_finetune.py -e baseline_finetune_resnet

# Train DenseNet on TT100K dataset
# 1. From scratch
python train.py -c config/tt100k_densenet_baseline.py -e baseline_densenet

# Meta-parameter tuning
# TODO

