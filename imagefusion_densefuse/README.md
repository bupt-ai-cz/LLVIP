# Densefuse: A Fusion Approach to Infrared and Visible Images - Tensorflow

[Hui Li](https://hli1221.github.io/), Xiao-Jun Wu*  
Published in: IEEE Transactions on Image Processing

*H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.*

- [IEEEXplore](https://ieeexplore.ieee.org/document/8580578)
- [arXiv](https://arxiv.org/abs/1804.08361)


## Note
In 'main.py' file, you will find how to run these codes.

The evaluate methods which used in our paper are shown in 'analysis_MatLab'. And these methods are implemented by MatLab.

## Abstract
In this paper, we present a novel deep learning architecture for infrared and visible images fusion problem. 

In contrast to conventional convolutional networks, our encoding network is combined by convolutional neural network layer and dense block which the output of each layer is connected to every other layer. We attempt to use this architecture to get more useful features from source images in encoder process. Then appropriate fusion strategy is utilized to fuse these features. Finally, the fused image is reconstructed by decoder. 

Compare with existing fusion methods, the proposed fusion method achieves state-of-the-art performance in objective and subjective assessment.

### The framework of fusion method
![](https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/framework.png)

### Fusion strategy - addition
<img src="https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/fuse_addition.png" width="600">

### Fusion strategy - l1-norm
<img src="https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/fuse_l1norm.png" width="600">


## Training

![](https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/train.png)

We train our network using [MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip)(T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) as input images which contains 80000 images and all resize to 256×256 and RGB images are transformed to gray ones. Learning rate is 1×10^(-4). The batch size and epochs are 2 and 4, respectively. Our method is implemented with GTX 1080Ti and 64GB RAM.


## Experimental results

### Infrared and visible images('street')
![](https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/fused_street.png)

### Infrared and visible images(RGB)
Database:  
Hwang S, Park J, Kim N, et al. Multispectral pedestrian detection: Benchmark dataset and baseline[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 1037-1045.  

![](https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/ivrgb_results.png)

### Multi-focus images(RGB)
![](https://github.com/hli1221/imagefusion_densefuse/blob/master/figures/fused_color.png)

If you have any question about this code, feel free to reach me(hui_li_jnu@163.com, lihui@stu.jiangnan.edu.cn)


# Citation

 *H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May 2019.*

```
@article{li2018densefuse,
  title={DenseFuse: A Fusion Approach to Infrared and Visible Images},
  author={Li, Hui and Wu, Xiao-Jun},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={5},
  pages={2614--2623},
  month={May},
  year={2019},
  publisher={IEEE}
}
```


## Pytorch version is available at [here](https://github.com/hli1221/densefuse-pytorch) (FOR REFERENCE ONLY)

