import os 
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--annotation_path', type=str, default='/root/LLVIP/Annotations', help='folder containing xml files')
    parser.add_argument('--image_path', type=str, default='/root/LLVIP/infrared/train', help='image path, e.g. /root/LLVIP/infrared/train')
    parser.add_argument('--txt_save_path', type=str, default='/root/yolov5/LLVIP/labels/train', help='txt path containing txt files in yolov5 format')
    opt = parser.parse_args()
    return opt
opt = parse_opt()

def convert_LLVIP_annotation(anno_path,image_path,txt_path):
    for i in tqdm(os.listdir(image_path)):
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

anno_path=opt.annotation_path
image_path=opt.image_path
txt_path=opt.txt_save_path
convert_LLVIP_annotation(anno_path,image_path,txt_path)