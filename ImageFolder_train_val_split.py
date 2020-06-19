import argparse
from torchvision.datasets import ImageFolder
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import sys
import math

def list_folders(path: str, include_path=False):
    if include_path:
        return [join(path,d) for d in listdir(path) if isdir(join(path,d))]    
    return [d for d in listdir(path) if isdir(join(path,d))]    

def list_files(path: str, include_path=False):
    if include_path:
        return [join(path,f) for f in listdir(path) if isfile(join(path,f))]
    return [f for f in listdir(path) if isfile(join(path,f))]

def write_list_to_file(filename: str, images: list):
    with open(filename, 'w') as f:
        for img in images:
            f.write(img + "\n")
    print("File {} written".format(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This script allows to generate a train/validation split of an ImageFolder-like structured dataset.")

    parser.add_argument("--input_folder", type=str, required=True, help="Path to directory containing dataset following ImageFolder structure")
    parser.add_argument("--val_size", type=float, default=0.1, help="Percentage of dataset to be selected for the validation set")
    parser.add_argument("--keep_class_ratio", action="store_true", help="To force the random sampling strategy to keep the original ratio in train/validation split")
    parser.add_argument("--train_split_output", type=str, default="train_split.txt", help="Name for output file containing the list of images in the training split")
    parser.add_argument("--val_split_output", type=str, default="val_split.txt", help="Name for output file containing the list of images in the validation split")

    args = parser.parse_args()

    assert isdir(args.input_folder), "The input folder does not exist!"
    assert args.val_size < 1 and args.val_size > 0, "Validation size is not correct. please give number between 0 and 1"

    classes = list_folders(args.input_folder)

    images = []
    images_dict = {}

    for cls in classes:
        cls_images = list_files(join(args.input_folder, cls), include_path=True)
        images.extend(cls_images)
        images_dict[cls] = np.array(cls_images)

    images = np.array(images)
    total_images = len(images)
    print("Total images: {}".format(len(images)))

    if not args.keep_class_ratio:
        val_split_size = int(args.val_size * total_images)
        print("Total val images: {}".format(val_split_size))

        val_indices = np.random.choice(total_images, val_split_size, replace=False)
        
        val_mask = np.zeros(total_images, dtype=bool)
        val_mask[val_indices] = True

        train_mask=~val_mask

        train_images = images[train_mask]
        val_images = images[val_mask]

        write_list_to_file(args.train_split_output, train_images)
        write_list_to_file(args.val_split_output, val_images)
        
        sys.exit(0)

    val_images = []
    train_images = []

    for cls in classes: 
        cls_images = images_dict[cls]
        total_cls_images = len(cls_images)

        if total_cls_images  <= 1:
            print("For class {} there's only one image! I am not putting it in the val split".format(cls))
            continue
        val_cls_images = int(math.ceil(args.val_size * total_cls_images))

        print("For class {} val images: {}".format(cls, val_cls_images))

        val_indices = np.random.choice(total_cls_images, val_cls_images, replace=False)
        val_mask = np.zeros(total_cls_images, dtype=bool)
        val_mask[val_indices] = True
        train_mask=~val_mask

        val_images.extend(cls_images[val_mask].tolist())
        train_images.extend(cls_images[train_mask].tolist())
    print("Total val images: {}".format(len(val_images)))

    write_list_to_file(args.train_split_output, train_images)
    write_list_to_file(args.val_split_output, val_images)

