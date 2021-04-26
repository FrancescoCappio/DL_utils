# This script is designed to work with detections.pkl output files generated
# by https://github.com/jwyang/faster-rcnn.pytorch and similar implementations

import argparse
import os 
parser = argparse.ArgumentParser(description="Script to export images with bounding box from detections")

parser.add_argument("--dets_file", type=str, help="Path for detections pkl file", required=True)
parser.add_argument("--dataset", type=str, help="Name for dataset in VOC format", required=True)
parser.add_argument("--split_file", type=str, help="Split to select (usually 'test')", default='test')
parser.add_argument("--output_dir", type=str, help="Output", default='output_bb')

args = parser.parse_args()

filepath = args.dets_file

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

import pickle
dets = pickle.load(open(filepath, "rb"))

imageset = "datasets/" + args.dataset + "/ImageSets/Main/{}.txt".format(args.split_file)

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

cmap = plt.cm.get_cmap('hsv', 21)
fnt = ImageFont.truetype(os.path.expanduser('~/FreeMonoBold.ttf'), 35)

from tqdm import tqdm
for img_idx, img_id in enumerate(tqdm(files_set)):
    #print(img_idx, ": ", img_id)
    img_dets_classes = []
    img_dets_boxes = []
    for class_idx, class_dets in enumerate(dets):
        if class_idx == 0:
            # background
            continue
        img_class_dets = class_dets[img_idx]
        n_class_dets = len(img_class_dets)
        n_class_preds = 0
        
        for det in img_class_dets:
            xmin,ymin,xmax,ymax,conf=det
            if conf > 0.5:
                n_class_preds += 1
                img_dets_classes.append(class_idx)
                img_dets_boxes.append((xmin,ymin,xmax,ymax,conf))
        #print("For class {} detections count: {}, preds count: {}".format(class_idx, n_class_dets, n_class_preds))
        
    img = Image.open("datasets/" + args.dataset + "/JPEGImages/{}.jpg".format(img_id)).convert("RGB")
    
    min_size = 600
    w,h = img.size

    if w <= h:
        nw = 600
        new_old_ratio = nw/w
        nh = int(h*new_old_ratio)
    else:
        nh = 600
        new_old_ratio = nh/h
        nw = int(w*new_old_ratio)

    for idx, box in enumerate(img_dets_boxes):
        xmin,ymin,xmax,ymax,conf = box
        
        xmin *= new_old_ratio
        ymin *= new_old_ratio
        xmax *= new_old_ratio
        ymax *= new_old_ratio
        box = (xmin,ymin,xmax,ymax,conf)
        img_dets_boxes[idx] = box


    img = img.resize((nw,nh), Image.ANTIALIAS)

    draw = Draw(img)
    
    for box, label in zip(img_dets_boxes, img_dets_classes):
        xmin,ymin,xmax,ymax,conf = box
        color = tuple([int(c * 255) for c in cmap(label)[:3]]) 
        draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(color), width=10)
        mtext = "{:.2f}".format(conf)

        draw.rectangle([xmin, ymin, xmin+100, ymin+40], outline=tuple(color), fill=tuple(color), width=10)
        left_off=7
        top_off=5
        draw.text((xmin-1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin-1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin+1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin+top_off), mtext, font=fnt, fill=(255,255,255,128))

    img.save(args.output_dir + "/{}.jpg".format(img_id) , "JPEG", quality=80, optimize=True, progressive=True)
    del draw
    
