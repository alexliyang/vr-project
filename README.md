# Scene Understanding For Autonomous Driving

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


## Week2 
We have been training and finetunning VGG, ResNet and DenseNet to be able to use it in an image classification problem. Results and comparisons will be written in the report.

### Code:
  - `models/denseNet_FCN.py` - implementation taken from [here](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) and adapted to our framework. Also, we made a correction with the axis of the batch normalization layers for TensorFlow.
  - `models/resnet.py` - implementation of ResNet using the Keras model, adapted to our framework.
  - `analyze_datasets.py`
  - `optimization.py` - optimization of a model by doing grid search, it contains a function that automatically generates the config files for each grid search iteration.

  #### How to use the code: 
  
    In `vr_project/code` directory:
    
     VGG baseline config: `python train.py -c config/tt100k_classif.py -e baseline_vgg`
      
     Resize to 256, 256 and then take random crops of 224, 224: `python train.py -c config/tt100k_classif_crop.py -e crop_vgg` 
      
    Substract mean and divide by std computed on the train set as image preprocessing 
      `python train.py -c config/tt100k_classif_preprocess.py -e preprocess_vgg`   

     Train from scratch ResNet:
      `python train.py -c config/tt100k_resnet_baseline.py -e baseline_resnet`  
    
     Fine-tune on ImageNet weights: 
      `python train.py -c config/tt100k_resnet_baseline_finetune.py -e baseline_finetune_resnet`  
    
     Train DenseNet on TT100K dataset:   
      `python train.py -c config/tt100k_densenet_baseline.py -e baseline_densenet`
    
### Results:

1. VGG - TODO
2. ResNet - TODO
3. DenseNet - TODO
4. Optimization - TODO

### GOALS: 
  1. **VGG**:
    - [x] - Analyze dataset - We extracted a CSV, statistical conclusions and plots of the classes distributions in the dataset(TT100K_TrafficSigns). Plots and comments in the report.
    - [x] - Train from scratch using TT100K.
    - [x] - Comparison between crop and resize.
    - [x] - Evaluate different pre-processings in the configuration file: subtracting mean and std feature-wise.
     - [ ] \(**DATASET MISSING**) - Transfer learning from TT100k dataset to Belgium dataset
     - [ ]  \(**DATASET MISSING**)- Train from scratch or finetune (or both) VGG with KITTI dataset
   2. **ResNet**:
     - [x] - Implement it and adapt it to the framework
     - [x] - Train from scratch with TT100K dataset
     - [x] - Finetunning from ImageNet weights with TT100K dataset
     - [x] - Compare finetunning vs train from scratch 
   3. **DenseNet**:
     - [x] - Implement it and adapt it to the framework
     - [ ] - Train from scratch with TT100K dataset
     
   4. **Boost performance** 
      - In progress 
      
   5. **Report** 
   
      - In progress 
      
   
## Report (_in progress_)

A detailed report about the work done can be found in [this](https://www.overleaf.com/read/nfmcpnydkwhb) Overleaf project. 

Additionally, a Google Slides presentation can be found in [this link](https://drive.google.com/open?id=1HpHPrQAMaI4yfxdcumAXnMNNF04tiprdRPl3zCxhUb8).


## DNN weights
HDF5 weights of the trained deep neural networks can be found 
[here](https://drive.google.com/open?id=0ByrI9_WaU23FdHoxX1h4X2ZXYUU).

## References

[1] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale
image recognition,” CoRR, vol. abs/1409.1556, 2014. **[SUMMARY](https://drive.google.com/open?id=0B8Ql6cxgb4lXc0FWWHAyVWVoYU0)**

[2] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke,
and A. Rabinovich, “Going deeper with convolutions,” in Computer Vision
and Pattern Recognition (CVPR), 2015. 

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,”
in arXiv prepring arXiv:1506.01497, 2015. **[SUMMARY](https://drive.google.com/open?id=0ByrI9_WaU23FQ042WDB1TTJvc1U)**
