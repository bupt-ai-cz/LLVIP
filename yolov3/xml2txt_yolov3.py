import os 
import argparse
import xml.etree.ElementTree as ET

def convert_LLVIP_annotation(anno_path,image_path,txt_path):
    #anno_path: the path to the .xml file.
    #image_path: the path to the .jpg file.
    #txt_path: the path to the .txt file，you need to create a .txt file in advance.
    with open(txt_path,'a') as f:
        for i in os.listdir(image_path):
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

anno_path='/root/LLVIP/Annotations'
image_path='/root/LLVIP/visible/train'
txt_path='/root/LLVIP/train.txt'
convert_LLVIP_annotation(anno_path,image_path,txt_path)
    