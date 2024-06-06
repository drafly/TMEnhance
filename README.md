# <font color=red>TMEnhance. </font> 

**2024.5.31:** Upload the HDR image enhancement code.

<br/>

## Abstract

High Dynamic Range (HDR) tone-mapping is essential in computer vision for accurately presenting HDR images on Low Dynamic Range (LDR) displays. Despite the significant progress made by deep neural networks (DNNs) in other areas of image processing, their application in HDR tone-mapping has been limited due to the absence of clear ground truth solutions for generating training data. To address this challenge, unsupervised methods, particularly those based on Generative Adversarial Networks (GANs), are often considered viable alternatives, reducing the need for paired datasets. These methods use a discriminator to distinguish the reconstructed image from the LDR reference image and optimize the generator's process of reconstructing the image through an adversarial loss. However, considering the significant differences in image content, this optimization mainly acts on effective constraints on the structural characteristics of the image. To complement the content feature constraints during image reconstruction, we introduce a content-preserving loss in the generator’s loss function. This strategy aims to ensure that the output LDR image faithfully reproduces the visual features of the HDR image. Specifically, we selectively integrate the tone mapping results as prior knowledge into the loss function of the generator to supervise the content features with different brightness distributions in the output image. To validate our method, we constructed an HDR-LDR dataset for training and conducted generalization experiments on two benchmark datasets. The experimental results demonstrate that our method excels in generating realistic and artifact-free tone-mapped images, achieving competitive performance across multiple image quality evaluation metrics. 

<br/>

## Usage:

### I. HDR Enhancement (HDR10-DIV2K dataset, 900 training image, 100 testing image)

1. The dataset is saved on Server 30 at "/data/dataset/xiaozhou/HDR10-DIV2K".  The dataset should contains 900 training image and 100 testing image, and should format like:

```
Your_Path
  -- our900
      -- hdr
      -- cond_hdr
      -- div2k
      -- linear
      -- youtube
      -- gamma
  -- eval100
      -- hdr
      -- cond_hdr
```

2. Evaluation pretrain model on HDR10-DIV2K dataset
```
python test_gan.py -opt options/test/test_GAN.yml
```

Results:
链接：https://pan.baidu.com/s/1tXUUNXvXj2TuigLFCxgaJQ   提取码：e8p7

3. Training your model on HDR10-DIV2K dataset.
```
python train_gan.py -opt options/train/train_GAN.yml
```

<br/>


