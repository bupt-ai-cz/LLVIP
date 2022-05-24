import os 
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--annotation_path', type=str, default='/root/LLVIP/Annotations', help='folder containing xml files')
    parser.add_argument('--image_path', type=str, default='/root/LLVIP/infrared/train', help='image path, e.g. /root/LLVIP/infrared/train')
    parser.add_argument('--txt_path', type=str, default='/root/yolov5/LLVIP/labels/train.txt', help='saving all bboxes of all images to a txt file')
    opt = parser.parse_args()
    return opt
opt = parse_opt()

def convert_LLVIP_annotation(anno_path,image_path,txt_path):
    #anno_path: the path to the .xml file.
    #image_path: the path to the .jpg file.
    #txt_path: the path to the .txt file，you need to create a .txt file in advance.
    with open(txt_path,'a') as f:
        for i in tqdm(os.listdir(image_path)):
            annotation=image_path+'/'+i
            root=ET.parse(anno_path+'/'+i.split(".")[0]+'.xml').getroot()
            objects = root.findall('object')
            for obj in objects:
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(0)])
            f.write(annotation + "\n")

anno_path=opt.annotation_path
image_path=opt.image_path
txt_path=opt.txt_path
convert_LLVIP_annotation(anno_path,image_path,txt_path)