# [Connection Reduction of DenseNet for Image Recognition](https://arxiv.org/abs/2208.01424)
<p align="center">
  <img src="Img/baseline.jpg" width="640" title="baseline">
</p>

## Citation
If you find ThreshNet useful in your research, please consider citing:

	@article{Connection_Reduction,
	 title={Connection Reduction of DenseNet for Image Recognition},
	 author={Rui-Yang Ju, Jen-Shiun Chiang},
	 conference={ISPACS},
	 year={2022}
	 }
	 
## Contents
1. [Introduction](#introduction)
2. [Usage](#Usage)
3. [Results](#Results)
4. [Requirements](#Requirements)
5. [Config](#Config)
6. [References](#References)

## Introduction
Convolutional Neural Networks (CNN) increase depth by stacking convolutional layers, and deeper network models perform better in image recognition. Empirical research shows that simply stacking convolutional layers does not make the network train better, and skip connection (residual learning) can improve network model performance. For the image classification task, models with global densely connected architectures perform well in large datasets like ImageNet, but are not suitable for small datasets such as CIFAR-10 and SVHN. Different from dense connections, we propose two new algorithms to connect layers. Baseline is a densely connected network, and the networks connected by the two new algorithms are named ShortNet1 and ShortNet2 respectively. The experimental results of image classification on CIFAR-10 and SVHN show that ShortNet1 has a 5% lower test error rate and 25% faster inference time than Baseline. ShortNet2 speeds up inference time by 40% with less loss in test accuracy.

 <img src="Img/connection.jpg" width="480" title="connetion">

## Usage
```bash
python3 main.py
```
optional arguments:

    --lr                default=1e-3    learning rate
    --epoch             default=200     number of epochs tp train for
    --trainBatchSize    default=100     training batch size
    --testBatchSize     default=100     test batch size

## Results
| Name | C10 GPU Time (ms) | C10 Error (%) | SVHN GPU Time (ms) | SVHN Error (%) | FLOPs (G) | MAdd (G) | Memory (MB) | #Params (M) | MenR+W (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline43 | 72.83 | 14.00 | 72.64 | 5.95 | 509.38 | 1.02 | 6.08 | 2.17 | 25.93 |
| ShortNet1-43 | 61.17 | 13.59 | 58.97 | 5.65 | 374.00 | 0.75 | 4.60 | 1.59 | 18.92 |
| ShortNet2-43 | 52.48 | 14.09 | 50.61 | 5.48 | 256.44 | 0.51 | 4.00 | 0.97 | 13.74|
| Baseline53 | 94.25 | 13.38 | 92.11 | 5.92 | 783.20 | 1.56 | 7.37 | 3.15 | 35.46 |
| ShortNet1-53 | 71.19 | 13.36 | 69.57 | 5.63 | 536.76 | 1.07 | 5.41 | 2.16 | 24.56 |
| ShortNet2-53 | 58.14 | 14.08 | 55.34 | 6.59 | 334.76 | 0.67 | 4.37 | 1.20 | 16.05 |

\* GPU Time is the inference time per 100 images on NVIDIA RTX 3050
 
  <img src="Img/C10.png" width="480" title="C10">
  <img src="Img/SVHN.png" width="480" title="SVHN">

## Requirements
* Python 3.6+
* Pytorch 0.4.0+
* Pandas 0.23.4+
* NumPy 1.14.3+

## Config
###### Optimizer 
__Adam Optimizer__
###### Learning Rate
__1e-3__ for [1,74] epochs <br>
__5e-4__ for [75,149] epochs <br>
__2.5e-4__ for [150,200) epochs <br>


## References
* [torchstat](https://github.com/Swall0w/torchstat)
* [pytorch-cifar10](https://github.com/soapisnotfat/pytorch-cifar10)
