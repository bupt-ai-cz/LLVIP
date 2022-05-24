## Toolbox
Our toolbox is used for various format conversions (xml to yolov5, xml to yolov3, xml to coco)
### xml to yolov5
```
python xml2txt_yolov5.py --annotation_path path/to/annotations --image_path path/to/images --txt_save_path path/to/txt_save_dir
```
### xml to yolov3
```
python xml2txt_yolov3.py --annotation_path path/to/annotations --image_path path/to/images --txt_path path/to/txtfile
```
### xml to coco
```
python voc2coco.py --annotation_path path/to/annotations --json_save_path path/to/jsonfile
```
