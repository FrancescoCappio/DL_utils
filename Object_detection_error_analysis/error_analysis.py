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

def draw_bbox(image_path, boxes, is_image_file = False):
    if is_image_file == False:
        pil_image = Image.open(image_path)
    else:
        pil_image = image_path
    fnt = ImageFont.truetype('arial.ttf', 35)
    draw = Draw(pil_image)
    #for box, color, score in zip(boxes, colors, scores):
    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=(0,0,0))
        #draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=tuple(color), width=15)
        
        mtext = "{:.2f}".format(0.8020)
        left_off=7
        top_off=5
        xmin,ymin,xmax,ymax = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        #draw.rectangle([xmin, ymin, xmin+100, ymin+40], outline=(0,0,0), width=10)
        #draw.text((xmin-1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        #draw.text((xmin+left_off,ymin-1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        #draw.text((xmin+1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        #draw.text((xmin+left_off,ymin+1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        #draw.text((xmin+left_off,ymin+top_off), mtext, font=fnt, fill=(255,255,255,128))
        
    del draw

    return pil_image

def iou(boxList1, boxList2):
    #INTERSEZIONE

    lt = torch.max(boxList1[:, None, :2], boxList2[:, :2])  # [N,M,2]
    rb = torch.min(boxList1[:, None, 2:], boxList2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    ###############
    #AREA 
    area1 = (boxList1[:, 2] - boxList1[:, 0] + TO_REMOVE) * (boxList1[:, 3] - boxList1[:, 1] + TO_REMOVE)
    area2 = (boxList2[:, 2] - boxList2[:, 0] + TO_REMOVE) * (boxList2[:, 3] - boxList2[:, 1] + TO_REMOVE)
    #IoU
    iou = inter / (area1[:, None] + area2 - inter)
    
    return iou

"""
anno_path = "D:\Desktop\DATASETS\clipart\Annotations\\"
imm_path = "D:\Desktop\DATASETS\clipart\JPEGImages\\"

anno_files = listdir(anno_path)
imm_files = listdir(imm_path)

anno = ET.parse(anno_path+anno_files[0]).getroot()

res = preprocess_annotation(anno)
print(anno_files[0])
print(res)


box_image = draw_bbox(imm_path + imm_files[0], res["boxes"])

lista_box = [(45,35,106,101)]

new_bbox = torch.tensor(lista_box, dtype=torch.float32)

box_image2 = draw_bbox(box_image, new_bbox, True)

#box_image2.show()

iou = iou(new_bbox, res["boxes"])

print(iou)
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, help='Dataset root  dir')
parser.add_argument('--detections', type=str, help="Saved Detections .pth file")
parser.add_argument('--anno_path', type=str, help='Path to annotations .xml')
parser.add_argument('--n_most_conf', type=int, default=2000, help='Number of most confidence predictions to condider for the Error Analysis')
args = parser.parse_args()

classes = ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



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

    iou_res = iou(detections[i].bbox, immage_info["boxes"])

    iou_with_gt, gt_index = iou_res.max(axis=1)

    del iou_res


    for k in range(len(detections[i].extra_fields['labels'])):

        temp_dict = {}
        temp_dict[f"{i}_{k}"] = k
        temp_dict["label_p"] = classes[detections[i].extra_fields['labels'][k]]

        temp_dict["label_gt"] = immage_info["labels"][gt_index[k]]
        temp_dict["score"] = detections[i].extra_fields['scores'].numpy()[k]
        temp_dict["iou_gt"] = iou_with_gt.numpy()[k]

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

