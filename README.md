# Deep Understanding of Traffic Scenes for Autonomous Driving

In this project we leverage state-of-the-art deep neural networks architectures for image classification,
object recognition and semantic segmentation to implement a framework that aids autonomous
driving by understanding the vehicle's surrounding scene.

## Authors

#### _V1P - V1sion emPowering_  

| Name | E-mail | GitHub |
| :---: | :---: | :---: |
| Arantxa Casanova | ar.casanova.8@gmail.com | [ArantxaCasanova](https://github.com/ArantxaCasanova) |
| Belén Luque | luquelopez.belen@gmail.com | [bluque](https://github.com/bluque) |
| Anna Martí | annamartiaguilera@gmail.com | [amartia](https://github.com/amartia) |
| Santi Puch | santiago.puch.giner@gmail.com | [santipuch590](https://github.com/santipuch590) |


## Report (_in progress_)

A detailed report about the work done can be found in [this](https://www.overleaf.com/read/nfmcpnydkwhb) Overleaf project. 

Additionally, a Google Slides presentation can be found in [this link](https://drive.google.com/open?id=1HpHPrQAMaI4yfxdcumAXnMNNF04tiprdRPl3zCxhUb8).


## DNN weights
HDF5 weights of the trained deep neural networks can be found 
[here](https://drive.google.com/open?id=0ByrI9_WaU23FdHoxX1h4X2ZXYUU).


## Datasets analysis
Prior to all experiments for each problem type (classification, detection and segmentation) we have performed an [analysis](https://drive.google.com/open?id=1X12gU2ey36rb43kPksHG0TC4MICftWRa7zByaTK6Egg) of the datasets to facilitate the interpretation of the results obtained.


## How to use the code
  
In `vr_project/code` directory:

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
      
    
#### Utilities

- Analyze datasets
    
    ```
    python analyze_datasets.py problem_type /path/to/datasets --output=/path/to/output/folder
    ```
    where `problem_type` must be either 'classification', 'detection' or 'segmentation'.

____

### Object recognition

In order to choose a good-performing object recognition network for our system, we have tested several CNNs with different architectures: VGG (2014), ResNet (2015) and DenseNet (2016). These networks have been both trained from scratch and fine-tuned using some pre-trained weights. The experiments have been carried out using different datasets: [TT100K classsification dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/) and [BelgiumTS dataset](http://btsd.ethz.ch/shareddata/) for traffic sign detection, and [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/raw_data.php) for cars, trucks, cyclists and other typical elements in driving scenes. Finally, we have tuned several parameters of the architectures and the training process in order to get better results. 

#### Contributions to the code

  - `models/denseNet_FCN.py` - adaptation of [this](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet)     implementation of DenseNet to the framework and generalization of the axes of the batch normalization layers, which was only working correctly for Theano.
  
  - `models/resnet.py` - adaptation of the resnet50 Keras model to the framework and included L2 regularization for the weights (not included in Keras Applications)
  
  - `models/vgg.py` - changed implementation to include L2 regularization for the weights (not included in Keras Applications)
  
  - `callbacks/callbacks.py` and `callbacks/callbacks_factory.py` - implemented a new callback, LRDecayScheduler, that allows the user to decay the learning rate by a predefined factor (such that lr <-- lr / decay_factor) at specific epochs, or alternatively at all epochs.
  
  - `analyze_datasets.py` - analyzes all the datasets in the specified folder by counting the number of images per class per set (train, validation, test), and creates a CSV file with the results and a plot of the (normalized) distribution for all sets.
  
  - `optimization.py` - automatically generates the config files for the optimization of a model, using a grid search, and launches the experiments.
  
  - `run_all.sh` - bash script to launch all the experiments in this project, including object recognition, object detection and semantic segmentation.
  
#### Milestones

1. **VGG**:
  - [x] Analyze dataset - We extracted a CSV, statistical conclusions and plots of the classes distributions in the dataset(TT100K_TrafficSigns). Plots and comments in the report.
  - [x] Train from scratch using TT100K.
  - [x] Comparison between crop and resize.
  - [x] Evaluate different pre-processings in the configuration file: subtracting mean and std feature-wise.
  - [x] Transfer learning from TT100k dataset to Belgium dataset
  - [x] Train from scratch and fine-tune VGG with KITTI dataset
2. **ResNet**:
  - [x] Implement it and adapt it to the framework
  - [x] Train from scratch with TT100K dataset
  - [x] Fine-tuning from ImageNet weights with the TT100K dataset
  - [x] Fine-tuning from ImageNet weights with the KITTI dataset
  - [x] Compare fine-tuning vs train from scratch 
3. **DenseNet**:
  - [x] Implement it and adapt it to the framework
  - [x] Train from scratch with TT100K dataset     
4. **Boost performance** 
  - [x] Grid-search to search hyperparams for ResNet
  - [x] Refined ResNet fine-tuning over ImageNet weights to boost the performance on TT100K dataset
  - [x] Implemented LR decay scheduler, that has proved to be helpful in improving the performance of the networks
  - [x] Try data augmentation and different parameters on DenseNet   
  
  
### Object detection  

For object detection we have considered two single-shot models: the most recent version of You Only Look Once (YOLO) together with its smaller counterpart, Tiny-YOLO, and Single-Shot Multibox Detector (SSD). The first two have been trained by fine-tuning the pre-trained ImageNet weights, while the latter has been trained from scratch. All these models have been trained to detect a variety of traffic signs in the [TT100K detection dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/) and to detect pedestrians, cars and trucks in the [Udacity](https://github.com/udacity/self-driving-car) dataset.

#### Contributions to the code

  - `models/ssd300.py`, `ssd_utils.py` and `metrics.py` - adaptation of [this](https://github.com/rykov8/ssd_keras) implementation of SSD300 to the framework, including the loss and batch generator utilities required to train it.

  - `analyze_datasets.py` - extended functionality to analyze detection datasets and report distributions over several variables. 
  
  - `eval_detection_fscore.py` - extended to evaluate the SSD model. Included options to control the detection and NMS thersholds. Added option to store the predictions for the first image in each processed chunk. Generalized the script to ignore specific classes, so that they are not taken into account when computing the metrics.
  
#### Milestones

1. **YOLO**:
  - [x] Fine-tune from ImageNet weights on TT100K detection dataset
  - [x] Fine-tune from ImageNet weights on Udacity dataset
  - [x] Evaluate performance on TT100K detection dataset
  - [x] Evaluate performance on Udacity dataset
2. **Tiny YOLO**:
  - [x] Fine-tune from ImageNet weights on TT100K detection dataset
  - [x] Fine-tune from ImageNet weights on Udacity dataset
  - [x] Evaluate performance on TT100K detection dataset
  - [x] Evaluate performance on Udacity dataset
  - [x] Compare results and performance  between TinyYOLO and YOLO
3. **SSD**:
  - [x] Implement it and adapt it to the framework
  - [x] Train from scratch on TT100K detection dataset
  - [x] Train from scratch on Udacity dataset
  - [x] Evaluate performance on TT100K detection dataset
  - [x] Evaluate performance on Udacity dataset
4. **Dataset Analysis**
  - [x] Analyze TT100K detection dataset: distribution of classes, bounding boxes' aspect ratios and bounding boxes' areas per dataset split
  - [x] Analyze Udacity dataset: distribution of classes, bounding boxes' aspect ratios and bounding boxes' areas per dataset split
  - [x] Assess similarities and differences between splits on Udacity dataset
4. **Boost performance** 
  - [x] Fine-tune Tiny YOLO from baseline weights on TT100K detection
  - [ ] Fine-tune Tiny YOLO and use preprocessing and data augmentation techniques to overcome the differences in dataset splits in Udacity dataset, thus improving the performance of the model on this dataset

### Experimental results

Prior to choosing our final system we have carried out several experiments using different architectures, different parameters and different datasets. A summary of the experiments done can be found [here](https://drive.google.com/open?id=1Qs51OxIPNOgOyujp98msk7RK1Rh2dkjfAHGZqUgBbqk).

____

## References

[1] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba. End to End Learning for Self-Driving Cars. arXiv:1604.07316 [cs], Apr. 2016. arXiv: 1604.07316.

[2] C. Chen, A. Seff, A. Kornhauser, and J. Xiao. Deepdriving: Learning affordance for direct perception in autonomous driving. In The IEEE International Conference on Computer Vision (ICCV), December 2015.

[3] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[4] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015. **[SUMMARY](https://drive.google.com/open?id=0ByrI9_WaU23FQ042WDB1TTJvc1U)**

[5] G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten. Densely Connected Convolutional Networks. Aug. 2016. arXiv: 1608.06993.

[6] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.

[7] M. Mathias, R. Timofte, R. Benenson, and L. Van Gool. Traffic sign recognition – how far are we from the solution? International Joint Conference on Neural Networks (IJCNN), 2013.

[8] J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. CoRR, abs/1506.02640, 2015. **[SUMMARY](https://drive.google.com/open?id=0ByrI9_WaU23FSlpkeEdGeDN3SlE)**

[9] J. Redmon and A. Farhadi. YOLO9000: better, faster, stronger. CoRR, abs/1612.08242, 2016.

[10] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. Sept. 2014. arXiv: 1409.0575.

[11] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. **[SUMMARY](https://drive.google.com/open?id=0B8Ql6cxgb4lXc0FWWHAyVWVoYU0)**

[12] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Computer Vision and Pattern Recognition (CVPR), 2015.

[13] Z. Zhu, D. Liang, S. Zhang, X. Huang, B. Li, and S. Hu. Traffic-sign detection and classification in the wild. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.
