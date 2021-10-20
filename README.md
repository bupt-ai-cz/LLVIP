# LLVIP: A Visible-infrared Paired Dataset for Low-light Vision
[Project](https://bupt-ai-cz.github.io/LLVIP/) | [Arxiv](https://arxiv.org/abs/2108.10831) | [Benchmarks](https://paperswithcode.com/dataset/llvip)|[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/llvip-a-visible-infrared-paired-dataset-for/pedestrian-detection-on-llvip)](https://paperswithcode.com/sota/pedestrian-detection-on-llvip?p=llvip-a-visible-infrared-paired-dataset-for) | [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"LLVIP:%20A%20Visible-infrared%20Paired%20Dataset%20for%20Low-light%20Vision"%20&url=https://github.com/bupt-ai-cz/LLVIP)  

## News

- ⚡(2021-09-01): We have released the dataset, please visit [homepage](https://bupt-ai-cz.github.io/LLVIP/) for access to the dataset. (Note that we removed some low-quality images from the original dataset, and for this version there are 30976 images.)

---

![figure1-LR](imgs/figure1-LR.png)

---

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

<h2> <p align="center"> Image Fusion </p> </h2>  

Baselines
   - [GTF](https://github.com/jiayi-ma/GTF)
   - [FusionGAN](https://github.com/jiayi-ma/FusionGAN)
   - [Densefuse](https://github.com/hli1221/imagefusion_densefuse)
   - [IFCNN](https://github.com/uzeful/IFCNN)

<h2> <p align="center"> Pedestrian Detection </p> </h2> 

Baselines
   - [Yolov5](https://github.com/ultralytics/yolov5)
   - [Yolov3](https://github.com/ultralytics/yolov3)
### Start
Clone this repo, download LLVIP dataset from the [homepage](https://bupt-ai-cz.github.io/LLVIP/) and install the dependent environment for yolov3 and yolov5 separately.
```bash
git clone https://github.com/bupt-ai-cz/LLVIP.git
cd LLVIP
```

We use [Yolov3](https://github.com/YunYang1994/tensorflow-yolov3) and [Yolov5](https://docs.ultralytics.com/tutorials/train-custom-datasets/) as baseline. Python>=3.8 is required.

### Yolov3:
1. Install requirements

    ```python
    cd yolov3
    pip install -r ./docs/requirements.txt
    ```

2. Train yolov3 on LLVIP dataset

  Three files are required as follows:
  - `./data/dataset/LLVIP_train.txt`
  - `./data/dataset/LLVIP_test.txt`
    
    ```
    xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
    xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
    # image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
    # make sure that x_max < width and y_max < height
    ```
    We provide a script named `xml2txt_yolov3.py` to convert xml file to txt file, remember to modify the file path before using.

     Then train from COCO weights:
       ```bash
       cd checkpoint
       wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
       tar -xvf yolov3_coco.tar.gz
       cd ..
       python convert_weight.py --train_from_coco
       python train.py
       ```
     The trained model will be saved in `./checkpoint` folder.

3. Evaluate on LLVIP dataset.

  To apply your trained model, edit your `./core/config.py` as follows:
  ```
  __C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=4.7528.ckpt-50"
                                    #replace here with your trained model.
  ```

  Then calculate mAP:
  ```bash
  python evaluate.py
  cd mAP
  python main.py -na
  ```
 ### Yolov5:
1. install requirements
    ```bash
    cd yolov5
    pip install -r requirements.txt
    ```
2. Train yolov5 on LLVIP dataset
  - File structure
  ![file structure](https://user-images.githubusercontent.com/33684330/136551790-0f962b2e-83c4-4981-9d29-b7d780267a8d.jpeg)
    Put train data in train folder and test data in val folder as shown above. We have prepared the label file, no need to generate it yourself.

  - Train from yolov5l weights:
    ```bash
    python train.py --img 1280 --batch 8 --epochs 200 --data LLVIP.yaml --weights yolov5l.pt --name LLVIP_export
    ```
    See more training options in `train.py`. The trained model will be saved in `./runs/train/LLVIP_export/weights` folder.
3. Evaluate on LLVIP dataset.
  ```bash
  python val.py --data --img 1280 --weights last.pt --data LLVIP.yaml
  ```
  Remember to put the trained model in the same folder as `val.py`.
### Results
We retrained and tested Yolov5l and Yolov3 on the updated dataset (30976 images).
![AP](https://user-images.githubusercontent.com/33684330/138012320-3340bf17-481a-4d69-a8a9-fc7427055cf4.jpg)

Where AP means the average of AP at IoU threshold of 0.5 to 0.95, with an interval of 0.05.

![yolov5_yolov3](https://user-images.githubusercontent.com/33684330/134609510-0408375c-7f4e-458c-938c-dd8c58c2248f.jpg)
The figure above shows the change of AP under different IoU thresholds. When the IoU threshold is higher than 0.7, the AP value drops rapidly. Besides, the infrared image highlights pedestrains and achieves a better effect than the visible image in the detection task, which not only proves the necessity of infrared images but also indicates that the performance of visible-image pedestrian detection algorithm is not good enough under low-light conditions.

We also drew the miss rate-FPPI curve based on the test results and calculated log average miss rate.
|log average miss rate |Yolov5l|Yolov3|
|:-----:|:-----|-----:|
|visible|22.59%|37.70%|
|infrared|10.66%|19.73%|

<div align="center">
<img src="https://user-images.githubusercontent.com/33684330/135218913-c0a6b668-b72b-4184-8b0f-b2ae37c9f6f6.jpg" height="380" width="800">
</div>

<h2> <p align="center"> Image-to-Image Translation </p> </h2> 

Baseline
   - [pix2pixGAN](https://github.com/phillipi/pix2pix)
### pix2pixGAN
1. install requirements
  ```bash
  cd pix2pixGAN
  pip install -r requirements.txt
  ```
2. train pix2pixGAN on LLVIP dataset.
  - [Prepare dataset](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md)

  - Train script
  ```bash
  python train.py --dataroot ./path/to/data --name LLVIP --model pix2pix --direction AtoB --batch_size 8 --preprocess scale_width_and_crop --load_size 320 --      crop_size 256 --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100
  ```
3. Test pix2pixGAN on LLVIP dataset.
  ```bash
  python test.py --dataroot ./datasets/LLVIP --name LLVIP --model pix2pix --direction AtoB --gpu_ids 0 --preprocess scale_width_and_crop --load_size 320 --crop_size 256
  ```
  See `./pix2pixGAN/options` for more train and test options.
### Results
We retrained and tested pix2pixGAN  on the updated dataset(30976 images). The structure of generator is unet256, and the structure of discriminator is the basic PatchGAN as default. 

|Dataset|SSIM|PSNR|
|:-----:|:--:|:--:|
|LLVIP|0.1757|10.7688|


<div align="center">
<img src="https://user-images.githubusercontent.com/33684330/135420925-72b9722a-3838-437b-b1a7-5f9e81c91d85.png" height="480" width="600">
</div>

## License
This LLVIP Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our [license terms](Term%20of%20Use%20and%20License.md).

## Call For Contributions

Welcome to point out errors in data annotation. Also welcome to contribute more data annotations, such as segmentation. Please contact us.

## Contact

email: shengjie.Liu@bupt.edu.cn, czhu@bupt.edu.cn, jiaxinyujxy@qq.com, tangwenqi@bupt.edu.cn
