# GridDehazeNet
This repo contains the official training and testing codes for our paper:

### GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
[Xiaohong Liu](https://xiaohongliu.ca)<sup>[*](#myfootnote1)</sup>, Mayong Rui<sup>[*](#myfootnote1)</sup>, Zhihao Shi, [Jun Chen](http://www.ece.mcmaster.ca/~junchen/)

<a name="myfootnote1">*</a> _Equal contribution_

Published on _2019 IEEE International Conference on Computer Vision (ICCV)_

[[Paper](https://proteus1991.github.io/GridDehazeNet/resource/GridDehazeNet.pdf)] [[Project Page](https://proteus1991.github.io/GridDehazeNet/)]
___

## Prerequisites
- Python >= 3.6  
- [Pytorch](https://pytorch.org/) >= 1.0  
- Torchvision >= 0.2.2  
- Pillow >= 5.1.0  
- Numpy >= 1.14.3
- Scipy >= 1.1.0

## Introduction
- ```train.py``` and ```test.py``` are the entry codes for training and testing the GridDehazeNet.
- ```train_data.py``` and ```val_data.py``` are used to load the training and validation/testing datasets.
- ```model.py``` defines the model of GridDehazeNet, and ```residual_dense_block.py``` builds the [RDB](https://arxiv.org/abs/1802.08797) block.
- ```perceptual.py``` defines the network for [perceptual loss](https://arxiv.org/abs/1603.08155).
- ```utils.py``` contains all corresponding utilities.
- ```indoor_haze_best_3_6``` and ```outdoor_haze_best_3_6``` are the trained weights for indoor and outdoor in SOTS from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0?authuser=0), where 3 and 6 stand for the network rows and columns (please read our paper for more details).
- The ```./trainning_log/indoor_log.txt``` and ```./trainning_log/outdoor_log.txt``` record the logs.
- The testing hazy images are saved in ```./indoor_results/``` or ```./outdoor_results/``` according to the image category.
- The ```./data/``` folder stores the data for training and testing.

## Quick Start

### 1. Testing
Clone this repo in environment that satisfies the prerequisites

```bash
$ git clone https://github.com/proteus1991/GridDehazeNet.git
$ cd GridDehazeNet
```
Run ```test.py``` using default hyper-parameter settings. 
```bash
$ python3 test.py
```
If everything goes well, you will see the following messages shown in your bash

```
--- Hyper-parameters for testing ---
val_batch_size: 1
network_height: 3
network_width: 6
num_dense_layer: 4
growth_rate: 16
lambda_loss: 0.04
category: indoor
--- Testing starts! ---
val_psnr: 32.16, val_ssim: 0.9836
validation time is 113.5568
```
This is our testing results of SOTS indoor dataset. For SOTS outdoor dataset, run

```bash
$ python3 test.py -category outdoor
```

If you want to change the default settings (e.g. modifying the ```val_batch_size``` since you have multiple GPUs), simply run

```bash
$ python3 test.py -val_batch_size 2
```
It is exactly the same way to modify any other hyper-parameters as shown above. For more details about the meaning of each hyper-parameter, please run

```bash
$ python3 test.py -h
```

### 2. Training
To retrain or fine-tune the GridDehazeNet, first download the ITS (for indoor) and OTS (for outdoor) training datasets from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0?authuser=0).
Then, copy ```hazy``` and ```clear``` folders from downloaded ITS and OTS to ```./data/train/indoor/``` and ```./data/train/outdoor/```. Here we provide the indoor and outdoor training list in ```trainlist.txt``` for reproduction purpose. Also, we found some hazy images in training set are quite similar to the testing set (use the same ground-truth images but with different parameters to generate hazy images). For fairness, we carefully remove all of them from the training set and write the rest in ```trainlist.txt```.

If you hope to use your own training dataset, please follow the same folder structure in ```./data/train/```). More details can be found in ```train_data.py```.

After put the training dataset into the correct path, we can train the GridDehazeNet by simply running ```train.py``` using default settings.
Similar to the [testing step](#quick-start), if there is no error raised, you will see the following messages shown in your bash

```
--- Hyper-parameters for training ---
learning_rate: 0.001
crop_size: [240, 240]
train_batch_size: 18
val_batch_size: 1
network_height: 3
network_width: 6
num_dense_layer: 4
growth_rate: 16
lambda_loss: 0.04
category: indoor
--- weight loaded ---
Total_params: 958051
old_val_psnr: 32.16, old_val_ssim: 0.9836
Learning rate sets to 0.001.
Epoch: 0, Iteration: 0
Epoch: 0, Iteration: 100
...
```
Follow the instruction in [testing](#quick-start) to modify the default settings.

## Cite
If you use any part of this code, please kindly cite

```
@inproceedings{liuICCV2019GridDehazeNet,
title={GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing},
author={Liu, Xiaohong and Ma, Yongrui and Shi, Zhihao and Chen, Jun},
booktitle={ICCV},
year={2019}
}
```


