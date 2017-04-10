# How to run the code?

#### Object recognition

  - VGG 
 
    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif.py -e baseline_vgg
    ```

    - Resize to 256, 256 and then take random crops of 224, 224 [TT100K dataset]
 
    ```
    python train.py -c config/tt100k_classif_crop.py -e crop_vgg
    ``` 

    - Substract mean and divide by std computed on the train set (as image preprocessing) [TT100K dataset]
 
    ```
    python train.py -c config/tt100k_classif_preprocess.py -e preprocess_vgg
    ```
    
    - Take random 224x224 crops and substract mean and divide by std computed on the train set [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_classif_crop_preprocess.py -e crop_preprocess_vgg
    ```
    
    - Transfer learning [TT100K dataset --> BelgiumTSC]
    
    ```
    python train.py -c config/belgium_vgg_crop.py -e transfer_vgg_crop
    ```
    
    - Baseline [KITTI dataset]
    
    ```
    python train.py -c config/kitti_baseline_vgg.py -e baseline_vgg
    ```

    - Fine-tune on ImageNet weights [KITTI dataset]
    
    ```
    python train.py -c config/kitti_finetune_vgg.py -e finetune_vgg
    ```
  
  - ResNet

    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_resnet_baseline.py -e baseline_resnet
    ```  

    - Fine-tune on ImageNet weights [TT100K dataset]
 
    ```
    python train.py -c config/tt100k_resnet_baseline_finetune.py -e baseline_finetune_resnet
    ```  
    
    - Improved fine-tune on ImageNet weights [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_resnet_baseline_finetune_lowerLR.py -e baseline_finetune_opt_resnet
    ```
    
    - Baseline [KITTI dataset]
    
    ```
    python train.py -c config/kitti_resnet_baseline.py -e baseline_resnet
    ```

    - Fine-tune on ImageNet weights [KITTI dataset]
    
    ```
    python train.py -c config/kitti_resnet_finetune_imagenet.py -e finetune_resnet
    ```

- DenseNet 

    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_densenet_baseline.py -e baseline_densenet
    ```
    
    - Optimization [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_densenet_opt.py -e opt_densenet
    ```
    
    - Re-train DenseNet with best weights, changing optimizer to ADAM [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_densenet_opt_different_opt.py -e densenet_trying_different_opt
    ```

#### Object detection

- YOLO

    - Baseline [TT100K detection]
  
    ```
    python train.py -c config/tt100k_detection.py -e baseline_yolo
    ```

    - Baseline [Udacity]

    ```
    python train.py -c config/udacity_yolo_baseline.py -e baseline_yolo
    ```

    - Fine-tune baseline model [TT100K detection]

    ```
    python train.py -c config/tt100k_yolo_improvements.py -e finetune_yolo
    ```

- Tiny-YOLO

    - Baseline [TT100K detection]

    ```
    python train.py -c config/tt100k_tiny_yolo.py -e baseline_tiny_yolo
    ```

    - Baseline [Udacity]

    ```
    python train.py -c config/udacity_tiny_yolo_baseline.py -e baseline_tiny_yolo
    ```
    
    - Fine-tune from the baseline_tiny_yolo weights [TT100K detection]
    
    ```
    python train.py -c config/tt100k_tiny_yolo_improvements.py -e tiny_yolo_improvements
    ```
    
    - Data augmentation to try to overcome the unbalanced datasets problem [Udacity]
    
    ```
    python train.py -c config/udacity_tiny_yolo_da.py -e tiny_yolo_da
    ```
    
- SSD300

    - Baseline [TT100K detection]
    
    ```
    python train.py -c config/tt100k_ssd300.py -e baseline_ssd300
    ```

    - Baseline [Udacity]

    ```
    python train.py -c config/udacity_ssd300.py -e baseline_ssd300
    ```

- Evaluation
    
    ```
    python eval_detection_fscore.py model dataset weights_path test_folder --detection-threshold=det_thr --nms-threshold=nms_thr --display display_bool --ignore-class idx
    ```
    
where:

- model: 'yolo', 'tiny-yolo' or 'ssd'
      
- dataset: 'TT100k_detection' or 'Udacity'
      
- weights_path: path to the weights HDF5 file for this model
      
- test_folder: path to the folder containing the dataset split to be evaluated
      
- det_thr: minimum confidence value for a prediction to be considered [Optional, default value = 0.5]
      
- nms_thr: Non-Maxima Supression threshold [Optional, default value = 0.2]
      
- display_bool: true or false, whether to store an image with the predictions for each chunk of processed data [Optional, default value = False]
      
- idx: list with the indices of the classes to be ignored, that is, not taken into account as predictions [Optional, default value = None]


#### Semantic Segmentation

- FCN8
    
    - Baseline [Camvid]
    
    ```
    python train.py -c config/camvid_fcn.py -e fcn_baseline
    ```
    
    - Baseline [Cityscapes]
    
    ```
    python train.py -c config/cityscapes_fcn.py -e fcn_baseline
    ```
    
- DeepLabv2
    
    - Baseline with ADAM optimizer [Camvid]
    
    ```
    python train.py -c config/camvid_deeplabv2_adam.py -e deeplabv2_adam
    ```
    
    - Baseline with ADAM optimizer and pre-processing [Cityscapes]
    
    ```
    python train.py -c config/camvid_deeplabv2_adam_preprocessing.py -e deeplabv2_adam_preprocessing
    ```
    
- SegNet

     - Baseline [Camvid]
    
    ```
    python train.py -c config/camvid_segnet.py -e segnet_baseline_scratch
    ```
    
- Tiramisu

    - FCN103 configuration, full training (crops + fine-tune full images) [Camvid]
    
    ```
    python train.py -c config/camvid_tiramisu_fc103_enhanced.py -e tiramisu_fc103_enhanced
    python train.py -c config/camvid_tiramisu_fc103_enhanced_finetune.py -e tiramisu_fc103_enhanced_finetune
    ```
    
    - FCN67 configuration, full training (crops + fine-tune full images) [Camvid]
    
    ```
    python train.py -c config/camvid_tiramisu_fc67_enhanced.py -e tiramisu_fc67_enhanced
    python train.py -c config/camvid_tiramisu_fc67_enhanced_finetune.py -e tiramisu_fc67_enhanced_finetune
    ```
    
    - FCN56 configuration, full training (crops + fine-tune full images) [Camvid]
    
    ```
    python train.py -c config/camvid_tiramisu_fc56_enhanced.py -e tiramisu_fc56_enhanced
    python train.py -c config/camvid_tiramisu_fc56_enhanced_finetune.py -e tiramisu_fc56_enhanced_finetune
    ```
    
    
#### Utilities

- Analyze datasets
    
    ```
    python analyze_datasets.py problem_type /path/to/datasets --output=/path/to/output/folder
    ```
    where `problem_type` must be either 'classification', 'detection' or 'segmentation'.
