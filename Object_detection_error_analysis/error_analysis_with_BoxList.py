#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL.ImageDraw import Draw
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import pdb
from maskrcnn_benchmark.structures.bounding_box import BoxList

def preprocess_annotation(target):
    boxes = []
    gt_classes = []
    difficult_boxes = []
    TO_REMOVE = 1

    for obj in target.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        if difficult:
            continue
        name = obj.find("name").text.lower().strip()
        bb = obj.find("bndbox")
        # Make pixel indexes 0-based
        # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
        box = [
            bb.find("xmin").text,
            bb.find("ymin").text,
            bb.find("xmax").text,
            bb.find("ymax").text,
        ]
        bndbox = tuple(
            map(lambda x: x - TO_REMOVE, list(map(int, box)))
        )

        boxes.append(bndbox)
        gt_classes.append(name)
        difficult_boxes.append(difficult)

    size = target.find("size")
    im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

    res = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": gt_classes,
        "difficult": torch.tensor(difficult_boxes),
        "im_info": im_info,
    }
    return res

def boxlist_iou(boxList1, boxList2):
    #INTERSEZIONE

    if boxList1.size != boxList2.size:
       raise RuntimeError("boxlists should have the same image size, got {}, {}".format(boxList1, boxList2))

    boxList1 = boxList1.convert("xyxy")
    boxList2 = boxList2.convert("xyxy")

    area1 = boxList1.area()
    area2 = boxList2.area()

    box1, box2 = boxList1.bbox, boxList2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    ###############
    #AREA 
    #area1 = (boxList1[:, 2] - boxList1[:, 0] + TO_REMOVE) * (boxList1[:, 3] - boxList1[:, 1] + TO_REMOVE)
    #area2 = (boxList2[:, 2] - boxList2[:, 0] + TO_REMOVE) * (boxList2[:, 3] - boxList2[:, 1] + TO_REMOVE)
    #IoU
    iou = inter / (area1[:, None] + area2 - inter)
    
    return iou

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, help='Dataset root  dir')
parser.add_argument('--detections', type=str, help="Saved Detections .pth file")
parser.add_argument('--anno_path', type=str, help='Path to annotations .xml')
parser.add_argument('--n_most_conf', type=int, default=2000, help='Number of most confidence predictions to condider for the Error Analysis')
parser.add_argument('--subset_classes', nargs='+', help="List of classes to consider")
args = parser.parse_args()

voc_classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

cityscapes_classes = ["__background__ ","person","rider","car","truck","bus","train","motorcycle","bicycle"]


classes = voc_classes

if args.subset_classes is not None:
    sub_set_classes = args.subset_classes
else:
    sub_set_classes = classes

detections = torch.load(open(args.detections, 'rb'))

lines = open(os.path.join(args.target, 'ImageSets', 'Main', 'test.txt'), 'r').readlines()
lines = [l.strip() for l in lines]

assert len(lines) == len(detections)

all_info = [] 

for i in range(len(lines)): 

    if (detections[i].bbox.shape[0] == 0):
       #questo caso significa che per quella immagine non ci sono predizioni della rete, cioè FN)
       continue


    annotations = ET.parse(args.anno_path + lines[i] +'.xml').getroot()
    immage_info = preprocess_annotation(annotations)

    if (immage_info["boxes"].shape[0] == 0):
       #caso penso impossibile, cioè ground-truth image without bbox
       continue


    im_height, im_width = immage_info["im_info"]
    detections[i] = detections[i].resize((im_width, im_height))

    detections[i].bbox[:, 2:] += 1
    immage_info["boxes"][:, 2:] += 1

    iou_res = boxlist_iou(BoxList(detections[i].bbox.numpy(),(im_width, im_height)), BoxList(immage_info["boxes"].numpy(), (im_width, im_height))).numpy()

    gt_index = iou_res.argmax(axis=1)
    iou_with_gt = iou_res.max(axis=1)

    del iou_res


    for k in range(len(detections[i].extra_fields['labels'])):

        temp_dict = {}
        temp_dict[f"{i}_{k}"] = k
        temp_dict["label_p"] = classes[detections[i].extra_fields['labels'][k]]

        temp_dict["label_gt"] = immage_info["labels"][gt_index[k]]
        temp_dict["score"] = detections[i].extra_fields['scores'].numpy()[k]
        temp_dict["iou_gt"] = iou_with_gt[k]

        if temp_dict["label_gt"] in sub_set_classes:
            all_info.append(temp_dict)



def take_score(elem):
    return elem["score"]

all_info_sort = sorted(all_info, key=take_score, reverse=True)

#ERROR ANALYSIS

#prendo i primi 1000 most confidence predictions
n_most_conf = args.n_most_conf 
all_info_sort = all_info_sort[:n_most_conf]
#print(all_info_sort)

correct = 0
misloc = 0
backgr = 0
counter = 0

for el in all_info_sort:

   if el["label_p"] == el["label_gt"]:

       if el["iou_gt"] < 0.3:
          backgr += 1
       elif el["iou_gt"] >= 0.5:
          correct += 1
       else:
          misloc += 1
   else:
       backgr += 1

   counter += 1

print(f"Correct detections: {(correct/counter)*100:.2f}%")
print(f"Mislocalization Error: {(misloc/counter)*100:.2f}%")
print(f"Background Error: {(backgr/counter)*100:.2f}%")
print(counter)

