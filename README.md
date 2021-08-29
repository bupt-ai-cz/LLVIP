# LLVIP: A Visible-infrared Paired Dataset for Low-light Vision 
[Project](https://bupt-ai-cz.github.io/LLVIP/) | [Arxiv](https://arxiv.org/abs/2108.10831) | [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"LLVIP:%20A%20Visible-infrared%20Paired%20Dataset%20for%20Low-light%20Vision"%20&url=https://github.com/bupt-ai-cz/LLVIP)

![figure1-LR](imgs/figure1-LR.png)

## Abstract

It is very challenging for various visual tasks such as image fusion, pedestrian detection and image-to-image translation in low light conditions due to the loss of effective target areas. In this case, infrared and visible images can be used together to provide both rich detail information and effective target areas. In this paper, we present LLVIP, a visible-infrared paired dataset for low-light vision. This dataset contains 33672 images, or 16836 pairs, most of which were taken at very dark scenes, and all of the images are strictly aligned in time and space. Pedestrians in the dataset are labeled. We compare the dataset with other visible-infrared datasets and evaluate the performance of some popular visual algorithms including image fusion, pedestrian detection and image-to-image translation on the dataset. The experimental results demonstrate the complementary effect of fusion on image information, and find the deficiency of existing algorithms of the three visual tasks in very low-light conditions. We believe the LLVIP dataset will contribute to the community of computer vision by promoting image fusion, pedestrian detection and image-to-image translation in very low-light applications.

## Baselines

1. Image Fusion
   - [GTF](https://www.sciencedirect.com/science/article/pii/S156625351630001X?via%3Dihub)
   - [FusionGAN](https://www.sciencedirect.com/science/article/pii/S1566253518301143)
   - [Densefuse](https://arxiv.org/abs/1804.08361)
   - [IFCNN](https://www.sciencedirect.com/science/article/pii/S1566253518305505)
2. Pedestrian Detection
   - [Yolov5](https://github.com/ultralytics/yolov5)
   - [Yolov3](https://arxiv.org/abs/1804.02767)
3. Image-to-image Translation
   - [pix2pixGAN](https://arxiv.org/abs/1611.07004)


## Citation
If you use this data for your research, please cite our paper [LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://arxiv.org/abs/2108.10831):

```
@article{jia2021llvip,
  title={LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  journal={arXiv preprint arXiv:2108.10831},
  year={2021}
}
```

## License
This LLVIP Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms bellow:

1. That you include a reference to the LLVIP Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the LLVIP website.
2. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
3. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
4. That all rights not expressly granted to you are reserved by us.

## Contact

email: jiaxinyujxy@qq.com; czhu@bupt.edu.cn
