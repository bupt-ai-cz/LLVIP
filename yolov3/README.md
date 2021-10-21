## Yolov3
### Preparation
- Install requirement
  ```bash
    cd yolov3
    pip install -r ./docs/requirements.txt
  ```
- File structure
  ```
  yolov3
  ├── ...
  └──data
     └── dataset
         ├──train.txt
         └──test.txt
  ```
  We provide a script named `xml2txt_yolov3.py` to convert xml file to txt file, remember to modify the file path before using.
### Train 
```bash
cd checkpoint
wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
tar -xvf yolov3_coco.tar.gz
cd ..
python convert_weight.py --train_from_coco
python train.py
```
The trained model will be saved in `./checkpoint` folder.

### Test
To apply your trained model, edit your `./core/config.py` as follows:
```bash
  __C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=4.7528.ckpt-50"
                                    #replace here with your trained model.
```
```bash
python evaluate.py
cd mAP
python main.py -na
```