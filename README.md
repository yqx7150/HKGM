# HKGM
**Paper**: One-shot Generative Prior Learned from Hankel-k-space for Parallel Imaging Reconstruction https://arxiv.org/abs/2208.07181

**Authors**: Hong Peng, Chen Jiang, Yu Guan, Jing Cheng, Minghui Zhang, Dong Liang, Senior Member, IEEE, Qiegen Liu, Senior Member, IEEE

Date : August-21-2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

Magnetic resonance imaging serves as an essential tool for clinical diagnosis. However, it suffers from a long acquisition time. The utilization of deep learning, especially the deep generative models, offers aggressive acceleration and better reconstruction in magnetic resonance imaging. Nevertheless, learning the data distribution as prior knowledge and reconstructing the image from limited data remains challenging. In this work, we propose a novel Hankel-k-space generative model (HKGM), which can generate samples from a training set of as little as one k-space data. At the prior learning stage, we first construct a large Hankel matrix from k-space data, then extract multiple structured k- space patches from the large Hankel matrix to capture the internal distribution among different patches. Extracting patches from a Hankel matrix enables the generative model to be learned from redundant and low- rank data space. At the iterative reconstruction stage, it is observed that the desired solution obeys the learned prior knowledge. The intermediate reconstruction solution is updated by taking it as the input of the generative model. The updated result is then alternatively operated by imposing low-rank penalty on its Hankel matrix and data consistency constrain on the measurement data. Experimental results confirmed that the internal statistics of patches within a single k-space data carry enough information for learning a powerful generative model and provide state-of-the-art reconstruction.

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

