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


## Object recognition

In order to choose a good-performing object recognition network for our system, we have tested several CNNs with different architectures: VGG (2014), ResNet (2015) and DenseNet (2016). These networks have been both trained from scratch and fine-tuned using some pre-trained weights. The experiments have been carried out using different datasets: [TT100K classsification dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/) and [BelgiumTS dataset](http://btsd.ethz.ch/shareddata/) for traffic sign detection, and **TODO: rest of datasets used**. Prior to the training, we have made an [analysis](https://drive.google.com/open?id=1X12gU2ey36rb43kPksHG0TC4MICftWRa7zByaTK6Egg) of the datasets to facilitate the interpretation of the results obtained. Finally, we have tuned several parameters of the architectures and the training process in order to get better results. After all the experiments done, we have chosen the **name** network as the best performing one for our system, reaching an accuracy of **acc**% in the TT100K dataset and **acc**% in the **name** dataset **etc...**.

### Contributions to the code

  - `models/denseNet_FCN.py` - adaptation of [this](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet)     implementation of DenseNet to the framework and generalization of the axes of the batch normalization layers, which was only working correctly for Theano.
  
  - `models/resnet.py` - adaptation of the resnet50 Keras model to the framework.
  
  - `analyze_datasets.py` - analyzes all the datasets in the specified folder by counting the number of images per class per set (train, validation, test), and creates a CSV file with the results and a plot of the (normalized) distribution for all sets.
  
  - `optimization.py` - automatically generates the config files for the optimization of a model, using a grid search, and launches the experiments.
  
  - `run.sh` - bash script to launch experiments, specifying existent config files.
  
### How to use the code
  
In `vr_project/code` directory:

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

  - ResNet

    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_resnet_baseline.py -e baseline_resnet
    ```  

    - Fine-tune on ImageNet weights [TT100K dataset]
 
    ```
    python train.py -c config/tt100k_resnet_baseline_finetune.py -e baseline_finetune_resnet
    ```  

- DenseNet 

    - Baseline [TT100K dataset]
    
    ```
    python train.py -c config/tt100k_densenet_baseline.py -e baseline_densenet
    ```
    
### Experimental Results:

TODO: short explanation
[link](https://docs.google.com/presentation/d/1CoZ54Plh_OXtutMvwdqmTDQqI6u52EK-r81eFH1JN7M/edit?usp=sharing)

### Milestones

1. **VGG**:
  - [x] Analyze dataset - We extracted a CSV, statistical conclusions and plots of the classes distributions in the dataset(TT100K_TrafficSigns). Plots and comments in the report.
  - [x] Train from scratch using TT100K.
  - [x] Comparison between crop and resize.
  - [x] Evaluate different pre-processings in the configuration file: subtracting mean and std feature-wise.
  - [x] Transfer learning from TT100k dataset to Belgium dataset
  - [ ] \(**DATASET MISSING**) Train from scratch or finetune (or both) VGG with KITTI dataset
2. **ResNet**:
  - [x] Implement it and adapt it to the framework
  - [x] Train from scratch with TT100K dataset
  - [x] Finetunning from ImageNet weights with TT100K dataset
  - [x] Compare finetunning vs train from scratch 
3. **DenseNet**:
  - [x] Implement it and adapt it to the framework
  - [ ] Train from scratch with TT100K dataset     
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

[1] K. He, X. Zhang, S. Ren, and J. Sun. "Deep residual learning for image recognition". CoRR, abs/1512.03385, 2015. **[SUMMARY](https://drive.google.com/open?id=0ByrI9_WaU23FQ042WDB1TTJvc1U)**

[2] G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten. "Densely Connected Convolutional Networks". Aug. 2016.
arXiv: 1608.06993.

[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks". In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger,
editors, Advances in Neural Information Processing Systems
25, pages 1097–1105. Curran Associates, Inc., 2012.

[4] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. "ImageNet Large Scale Visual Recognition Challenge". Sept. 2014. arXiv: 1409.0575.

[5] K. Simonyan and A. Zisserman. "Very deep convolutional networks for large-scale image recognition". CoRR, abs/1409.1556, 2014.1 **[SUMMARY](https://drive.google.com/open?id=0B8Ql6cxgb4lXc0FWWHAyVWVoYU0)**
