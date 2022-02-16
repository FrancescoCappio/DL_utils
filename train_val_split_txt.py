import argparse
import os
import numpy as np

def write_lines(file, lines):
    with open(file, "w") as out_f:
        for el in lines:
            out_f.write(f'{el[0]} {el[1]}\n')

parser = argparse.ArgumentParser("Script to split list of labeled files in train and validation sets")

parser.add_argument('input', help="Path to input file, which contains lines with format 'file_path lbl'")
parser.add_argument('--val_size', help="Proportion of data in validation set, default=0.2", default=0.2)
parser.add_argument('--per_class', action='store_true', help="Keep the same class ratios")

args = parser.parse_args()

input_file = args.input

assert os.path.isfile(input_file)
np_lines = np.genfromtxt(input_file, dtype='unicode')

if args.per_class:
    class_dict = {}

    for el in np_lines:
        if not el[1] in class_dict:
            class_dict[el[1]] = []
        class_dict[el[1]].append(el)

    train_out = []
    val_out = []

    for k in class_dict.keys():
        np_cls = np.array(class_dict[k])
        np.random.shuffle(np_cls)
        val_size = int(args.val_size*len(np_cls))
        val_cls = np_cls[:val_size]
        train_cls = np_cls[val_size:]

        train_out.extend([el for el in train_cls])
        val_out.extend([el for el in val_cls])

else:
    np.random.shuffle(np_lines)
    val_size = int(args.val_size*len(np_lines))

    val_out = np_lines[:val_size]
    train_out = np_lines[val_size:]

dirname = os.path.dirname(input_file)
if len(dirname) == 0:
    dirname = '.'
file_name = os.path.basename(input_file).split('.')[0]
out_train_file = os.path.join(dirname, file_name + '_train.txt')
out_val_file = os.path.join(dirname, file_name + '_val.txt')

write_lines(out_train_file, train_out)
write_lines(out_val_file, val_out)
