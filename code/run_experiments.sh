#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.

# EVALUATE TINY YOLO IMPROVEMENTS
python eval_detection_fscore.py tiny-yolo Udacity /home/master/data/master/Experiments/Udacity/tiny_yolo_da/weights.hdf5 /data/module5/Datasets/detection/Udacity/test/ --display true

python eval_detection_fscore.py tiny-yolo Udacity /home/master/data/master/Experiments/Udacity/tiny_yolo_da/weights.hdf5 /data/module5/Datasets/detection/Udacity/valid/ --display true

python eval_detection_fscore.py tiny-yolo Udacity /home/master/data/master/Experiments/Udacity/tiny_yolo_da/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/

# SSD TRAIN SETS
# Evaluate baseline SSD on the training set (TT100K)
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/train/

# Evaluate baseline SSD on the training set (Udacity)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/

# Evaluate baseline SSD on the training set (Udacity, ignored truck class)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/ --ignore-class 2
