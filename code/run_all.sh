#!/bin/bash

# SCRIPT TO TRAIN AND TEST ALL MODELS USED FOR OBJECT RECOGNITION, OBJECT DETECTION AND SEMANTIC SEGMENTATION


# ------------------------------------------------------------------------------------------------ #
# ------------------------------------- OBJECT RECOGNITION --------------------------------------- #
# ------------------------------------------------------------------------------------------------ #


#####################
####### VGG #########
#####################

# Train VGG on TT100K dataset from scratch
# 1. Baseline configuration
python train.py -c config/tt100k_classif.py -e baseline_vgg
# 2. Resize to 256, 256 and then take random crops of 224, 224
python train.py -c config/tt100k_classif_crop.py -e crop_vgg
# 3. Substract mean and divide by std computed on the train set as image preprocessing
python train.py -c config/tt100k_classif_preprocess.py -e preprocess_vgg
# 4. Substract mean and divide by std computed on the train set as image preprocessing, and apply random crops
python train.py -c config/tt100k_classif_crop_preprocess.py -e crop_preprocess_vgg

# VGG transfer learning from TT100K to Belgium Traffic Signs
python train.py -c config/belgium_vgg_crop.py -e transfer_vgg_crop

# Train VGG on KITTI dataset
# 1. From scratch
python train.py -c config/kitti_baseline_vgg.py -e baseline_vgg
# 2. Fine-tune on ImageNet weights
python train.py -c config/kitti_finetune_vgg.py -e finetune_vgg


#####################
###### ResNet #######
#####################

# Train ResNet on TT100K dataset
# 1. From scratch
python train.py -c config/tt100k_resnet_baseline.py -e baseline_resnet
# 2. Fine-tune on ImageNet weights
python train.py -c config/tt100k_resnet_baseline_finetune.py -e baseline_finetune_resnet

# ResNet transfer learning from TT100K to BTS
python train.py -c config/belgium_resnet.py -e transfer_resnet

# Train ResNet on KITTI dataset
# 1. From scratch
python train.py -c config/kitti_resnet_baseline.py -e baseline_resnet
# 2. Fine-tune on ImageNet weights
python train.py -c config/kitti_resnet_finetune_imagenet.py -e finetune_resnet


#####################
##### DenseNet ######
#####################

# Train DenseNet on TT100K dataset from scratch
python train.py -c config/tt100k_densenet_baseline.py -e baseline_densenet


######################
#### Improvements ####
######################

# Improve ResNet performance when fine-tuning ImageNet weights on TT100K
python train.py -c config/tt100k_resnet_baseline_finetune_lowerLR.py -e baseline_finetune_opt_resnet

# Meta-parameter tuning for ResNet on TT100K
python optimization.py

# Train DenseNet with data augmentation and different parameters on TT100K
python train.py -c config/tt100k_densenet_opt.py -e opt_densenet

# Re-train DenseNet with best weights, changing optimizer to ADAM
python train.py -c config/tt100k_densenet_opt_different_opt.py -e alternative_opt_densenet



# ------------------------------------------------------------------------------------------------ #
# -------------------------------------- OBJECT DETECTION ---------------------------------------- #
# ------------------------------------------------------------------------------------------------ #


#####################
####### YOLO ########
#####################

# Train YOLO on TT100K for detection
python train.py -c config/tt100k_detection.py -e baseline_yolo

# Train YOLO on Udacity dataset
python train.py -c config/udacity_yolo_baseline.py -e baseline_yolo


#####################
##### TINY YOLO #####
#####################

# Train Tiny-YOLO on TT100K for detection
python train.py -c config/tt100k_tiny_yolo.py -e baseline_tiny_yolo

# Train Tiny-YOLO on Udacity dataset
python train.py -c config/udacity_tiny_yolo_baseline.py -e baseline_tiny_yolo


#####################
###### SSD300 #######
#####################

# Train SSD300 on TT100K detection dataset
python train.py -c config/tt100k_ssd300.py -e baseline_ssd300

# Train SSD300 on Udacity dataset
python train.py -c config/udacity_ssd300.py -e baseline_ssd300


######################
#### Improvements ####
######################

# Fine-tune Tiny-YOLO on TT100K for detection, from the baseline_tiny_yolo weights
python train.py -c config/tt100k_tiny_yolo_improvements.py -e tiny_yolo_improvements

# Train YOLO on TT100K for detection
python train.py -c config/tt100k_yolo_improvements.py -e finetune_yolo

# Tiny-YOLO and use data augmentation on Udacity to overcome the unbalanced datasets problem
python train.py -c config/udacity_tiny_yolo_da.py -e tiny_yolo_da



# ------------------------------------------------------------------------------------------------ #
# ------------------------------------ SEMANTIC SEGMENTATION ------------------------------------- #
# ------------------------------------------------------------------------------------------------ #


######################
######## FCN #########
######################

# Train on CamVid
python train.py -c config/camvid_fcn.py -e fcn_baseline

# Train on CityScapes
python train.py -c config/cityscapes_fcn.py -e fcn_baseline


######################
###### DEEPLAB #######
######################

#Train DeepLabv2 in CamVid using adam optimizer
python train.py -c config/camvid_deeplabv2_adam.py -e deeplabv2_adam

#Train DeepLabv2 in CamVid using adam optimizer and preprocessing
python train.py -c config/camvid_deeplabv2_adam_preprocessing.py -e deeplabv2_adam_preprocessing


######################
####### SEGNET #######
######################

# Train on CamVid
python train.py -c config/camvid_segnet.py -e segnet_baseline_scratch


######################
###### TIRAMISU ######
######################

# Full FCN103 training (crops + fine-tune full images) on CamvVid
python train.py -c config/camvid_tiramisu_fc103_enhanced.py -e tiramisu_fc103_enhanced
python train.py -c config/camvid_tiramisu_fc103_enhanced_finetune.py -e tiramisu_fc103_enhanced_finetune

# Full FCN67 training (crops + fine-tune full images) on CamvVid
python train.py -c config/camvid_tiramisu_fc67_enhanced.py -e tiramisu_fc67_enhanced
python train.py -c config/camvid_tiramisu_fc67_enhanced_finetune.py -e tiramisu_fc67_enhanced_finetune

# Full FCN56 training (crops + fine-tune full images) on CamvVid
python train.py -c config/camvid_tiramisu_fc56_enhanced.py -e tiramisu_fc56_enhanced
python train.py -c config/camvid_tiramisu_fc56_enhanced_finetune.py -e tiramisu_fc56_enhanced_finetune
