#!/bin/bash

# SCRIPT TO BATCH RUN EXPERIMENTS.
# WARNING: ITS CONTENT IS GOING TO FREQUENTLY CHANGE, AS WE ARE GOING TO CUSTOMIZE IT FOR EACH BATCH
# OF EXPERIMENTS WE WANT TO EXECUTE.


# TEST SETS
# Evaluate baseline SSD on the test set (TT100K)
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/test/ --display true

# Evaluate baseline SSD on the test set (Udacity)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/test/

# Evaluate baseline SSD on the test set (Udacity, ignored truck class)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/test/ --display true --ignore-class 2

# VALIDATION SETS
# Evaluate baseline SSD on the validation set (TT100K)
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/valid/ --display true

# Evaluate baseline SSD on the validation set (Udacity)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/valid/

# Evaluate baseline SSD on the validation set (Udacity, ignored truck class)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/valid/ --display true --ignore-class 2

# EXTRAS
# Train Yolo on Udacity dataset with Data Aumentation
python train.py -c config/udacity_tiny_yolo_da.py -e tiny_yolo_da

# TRAIN SETS
# Evaluate baseline SSD on the training set (TT100K)
python eval_detection_fscore.py ssd TT100K_detection /home/master/data/master/Experiments/TT100K_detection/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/TT100K_detection/train/

# Evaluate baseline SSD on the training set (Udacity)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/

# Evaluate baseline SSD on the training set (Udacity, ignored truck class)
python eval_detection_fscore.py ssd Udacity /home/master/data/master/Experiments/Udacity/baseline_ssd300_v2/weights.hdf5 /data/module5/Datasets/detection/Udacity/train/ --ignore-class 2
