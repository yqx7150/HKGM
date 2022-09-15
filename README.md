# HKGM
**Paper**: One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction https://arxiv.org/abs/2208.07181

**Authors**: Hong Peng, Chen Jiang, Yu Guan, Jing Cheng, Minghui Zhang, Dong Liang, Senior Member, IEEE, Qiegen Liu, Senior Member, IEEE

Date : August-21-2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

Magnetic resonance imaging serves as an essential tool for clinical diagnosis. However, it suffers from a long acquisition time. The utilization of deep learning, especially the deep generative models, offers aggressive acceleration and better reconstruction in magnetic resonance imaging. Nevertheless, learning the data distribution as prior knowledge and reconstructing the image from limited data remains challenging. In this work, we propose a novel Hankel-k-space generative model (HKGM), which can generate samples from a training set of as little as one k-space data. At the prior learning stage, we first construct a large Hankel matrix from k-space data, then extract multiple structured k- space patches from the large Hankel matrix to capture the internal distribution among different patches. Extracting patches from a Hankel matrix enables the generative model to be learned from redundant and low- rank data space. At the iterative reconstruction stage, it is observed that the desired solution obeys the learned prior knowledge. The intermediate reconstruction solution is updated by taking it as the input of the generative model. The updated result is then alternatively operated by imposing low-rank penalty on its Hankel matrix and data consistency constrain on the measurement data. Experimental results confirmed that the internal statistics of patches within a single k-space data carry enough information for learning a powerful generative model and provide state-of-the-art reconstruction.

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26
    
## Test Demo
```bash
python PCsampling_demo_parallel.py
```

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Google Drive] (https://drive.google.com/file/d/1UMULob7RG70X9ChI1UgwHb6Lt3FB6THC/view?usp=sharing) [Baidu cloud] (https://pan.baidu.com/s/1P1h7FEvz9FuH3ZE6NX2WMA?pwd=jlzu)

## Graphical representation
### Pipeline of the prior learning process and PI reconstruction procedure in HKGM
<div align="center"><img src="https://github.com/yqx7150/HKGM/blob/main/figure1.png" >  </div>

The training flow chart of HKGM. The training process mainly consists of three steps. Firstly, we construct a large Hankel matrix from k-space data. After that, we extract a lot of redundancy and low-rank patches to generate sufficient data samples. Finally, we feed these training patches to the network to capture the internal distribution at different patches.


<div align="center"><img src="https://github.com/yqx7150/HKGM/blob/main/figure2.png" >  </div>

The pipeline of the PI reconstruction procedure in HKGM. The iterative reconstruction process mainly consists of three steps. Firstly, we iteratively reconstruct objects from the trained network using a PC sampler on the input k-space data. After that, we construct Hankel matrix from the output of the network and apply low-rank penalty on it. Finally, we perform data consistency on the k-space data formed reversely from the matrix.

## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.

## Other Related Projects
  * Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S0925231221000990) [<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)
  
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)



