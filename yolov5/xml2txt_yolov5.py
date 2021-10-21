import os 
import argparse
import xml.etree.ElementTree as ET

def convert_LLVIP_annotation(anno_path,image_path,txt_path):
    for i in os.listdir(image_path):
        root=ET.parse(anno_path+'/'+i.split(".")[0]+'.xml').getroot()
        objects = root.findall('object')
        for obj in objects:
            annotation = ''
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text.strip())
            xmax = int(bbox.find('xmax').text.strip())
            ymin = int(bbox.find('ymin').text.strip())
            ymax = int(bbox.find('ymax').text.strip())
            x_center = str((0.5*(xmin + xmax))/1280)
            y_center = str((0.5*(ymin + ymax))/1024)
            width = str((xmax-xmin)/1280)
            height = str((ymax-ymin)/1024)
            annotation = annotation.join([str(0),' ',x_center,' ',y_center,' ',width,' ',height])
            with open(txt_path+'/'+i.split(".")[0]+'.txt','a') as f:
                f.write(annotation + "\n")

anno_path='/root/LLVIP/Annotations'
image_path='/root/LLVIP/infrared/train'
txt_path='/root/LLVIP/yolov5/LLVIP/labels/train'
convert_LLVIP_annotation(anno_path,image_path,txt_path)