# LLVIP: A Visible-infrared Paired Dataset for Low-light Vision
[Project](https://bupt-ai-cz.github.io/LLVIP/) | [Arxiv](https://arxiv.org/abs/2108.10831) | [Benchmarks](https://paperswithcode.com/dataset/llvip) | [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"LLVIP:%20A%20Visible-infrared%20Paired%20Dataset%20for%20Low-light%20Vision"%20&url=https://github.com/bupt-ai-cz/LLVIP)

## News


- ⚡(2021-09-22): We have retrained and tested Yolov5 and Yolov3 on the updated dataset (30976 images). The results are displayed in README.
- ⚡(2021-09-01): We have released the dataset, please visit [homepage](https://bupt-ai-cz.github.io/LLVIP/) for access to the dataset. (Note that we removed some low-quality images from the original dataset, and for this version there are 30976 images.)

---

![figure1-LR](imgs/figure1-LR.png)

## Abstract

It is very challenging for various visual tasks such as image fusion, pedestrian detection and image-to-image translation in low light conditions due to the loss of effective target areas. In this case, infrared and visible images can be used together to provide both rich detail information and effective target areas. In this paper, we present LLVIP, a visible-infrared paired dataset for low-light vision. This dataset contains 30976 images, or 15488 pairs, most of which were taken at very dark scenes, and all of the images are strictly aligned in time and space. Pedestrians in the dataset are labeled. We compare the dataset with other visible-infrared datasets and evaluate the performance of some popular visual algorithms including image fusion, pedestrian detection and image-to-image translation on the dataset. The experimental results demonstrate the complementary effect of fusion on image information, and find the deficiency of existing algorithms of the three visual tasks in very low-light conditions. We believe the LLVIP dataset will contribute to the community of computer vision by promoting image fusion, pedestrian detection and image-to-image translation in very low-light applications.

## Baselines

1. Image Fusion
   - [GTF](https://github.com/jiayi-ma/GTF)
   - [FusionGAN](https://github.com/jiayi-ma/FusionGAN)
   - [Densefuse](https://github.com/hli1221/imagefusion_densefuse)
   - [IFCNN](https://github.com/uzeful/IFCNN)
2. Pedestrian Detection
   - [Yolov5](https://github.com/ultralytics/yolov5)
   - [Yolov3](https://github.com/ultralytics/yolov3)
3. Image-to-image Translation
   - [pix2pixGAN](https://github.com/phillipi/pix2pix)

## Image Fusion Results （being updated）

## Pedestrian Detection Results （being updated）
We retrained and tested Yolov5l and Yolov3 on the updated dataset(30976 images).
|model |      |Yolov5l|      |      |Yolov3|      |
|------|:-----|------|-----|:-----|------|-----:|
||AP50|AP75|AP|AP50|AP75|AP|
|visible|0.908|0.564|0.527|0.871|0.455|0.466|
|infrared|0.965|0.764|0.670|0.940|0.661|0.582|

Where AP50 means the AP at IoU threshold of 0.5, AP75 means the AP at IoU threshold of 0.75, and AP means the average of AP at IoU threshold of 0.5 to 0.95, with an interval of 0.05.

![yolov5_yolov3](https://user-images.githubusercontent.com/33684330/134609510-0408375c-7f4e-458c-938c-dd8c58c2248f.jpg)
The figure above shows the change of AP under different IoU thresholds. When the IoU threshold is higher than 0.7, the AP value drops rapidly. Besides, the infrared image highlights pedestrains and achieves a better effect than the visible image in the detection task, which not only proves the necessity of infrared images but also indicates that the performance of pedestrian detection algorithm is not good enough under low-light conditions.



## Image-to-Image Translation Results （being updated）

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
This LLVIP Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our [license terms](Term%20of%20Use%20and%20License.md).

## Call For Contributions

Welcome to point out errors in data annotation. Also welcome to contribute more data annotations, such as segmentation. Please contact us.

## Contact

email: czhu@bupt.edu.cn, jiaxinyujxy@qq.com, tangwenqi@bupt.edu.cn, or bupt.ai.cz@gmail.com
