#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# Evaluate baseline SSD on the test set
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/test/ --display true

# Evaluate baseline SSD on the training set
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/train/

# Evaluate baseline SSD on the validation set
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/valid/ --display true

## Evaluate baseline SSD on the test set
#python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/test/
#
## Evaluate baseline SSD on the training set
#python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/
#
## Evaluate baseline SSD on the validation set
#python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/valid/

## Train Yolo on Udacity dataset with Data Aumentation
#python train.py -c config/udacity_tiny_yolo_da.py -e tiny_yolo_da