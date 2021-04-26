# Datasets should be in VOC like format
# VOC classes are alreasy present, other datasets classes should be added.

import argparse
import os 
parser = argparse.ArgumentParser(description="Script to export images with Ground Truth bounding boxes")

parser.add_argument("--dataset", type=str, help="Name for dataset in VOC format", required=True)
parser.add_argument("--split_file", type=str, help="Split to select (usually 'test')", default='test')
parser.add_argument("--output_dir", type=str, help="Output", default='output_bb')

args = parser.parse_args()

VOC_CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

cls = VOC_CLASSES
cls_dict = dict(zip(cls, range(len(cls))))

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

imageset = args.dataset + "/ImageSets/Main/{}.txt".format(args.split_file)

fd = open(imageset, "r")
line = fd.readline()
files_set = []
while line:
    files_set.append(line)
    line = fd.readline()
    
files_set = [fs.strip() for fs in files_set]
       
from PIL import Image, ImageFont
import numpy as np
from PIL.ImageDraw import Draw
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

cmap = plt.cm.get_cmap('hsv', 21)
fnt = ImageFont.truetype(os.path.expanduser('~/FreeMonoBold.ttf'), 35)

from tqdm import tqdm
for img_idx, img_id in enumerate(tqdm(files_set)):

    img = Image.open(args.dataset + "/JPEGImages/{}.jpg".format(img_id)).convert("RGB")
    labels = ET.parse(args.dataset + "/Annotations/{}.xml".format(img_id))

    root = labels.getroot()
    boxes = []
    for obj in root.findall("object"):
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        class_name = obj.find("name").text
        box = (xmin,ymin,xmax,ymax,class_name)
        boxes.append(box)


    draw = Draw(img)
    
    for box in boxes:
        xmin,ymin,xmax,ymax,class_name = box
        label = cls_dict[class_name]
        color = tuple([int(c * 255) for c in cmap(label)[:3]]) 
        draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(color), width=10)

    img.save(args.output_dir + "/{}.jpg".format(img_id) , "JPEG", quality=80, optimize=True, progressive=True)
    del draw
    
