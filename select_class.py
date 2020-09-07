# Purpose: iterate over xml file listed, select only those containing the object car and 
# discard all other objects
import xml.etree.ElementTree as ET
import os
import argparse


def get_list_xml_files(txt_list: str, files_dir: str):
    # returns list containing paths of xml files to be filtered
    with open(txt_list) as f:
        content = f.readlines()
    to_select = [x.strip() for x in content] 

    files = [os.path.join(files_dir, f + ".xml") for f in to_select]

    for f in files:
        assert os.path.isfile(f), "File {} does not exist, check your list!"
    return files

def get_output_xml(output_dir, input_xml):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    output_name = input_xml.split("/")[-1]
    return os.path.join(output_dir, output_name)

    
def check_class(f, class_name):
    # check if file f contains class 
    tree = ET.parse(f)
    root = tree.getroot()
   
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name == class_name:
            return True
    return False

def remove_other_classes(f, class_name):
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != class_name:
            root.remove(obj)
    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to filter and modify xml files selecting only one class")

    parser.add_argument("--source_list", type=str, required=True, help="Path to text file containing list of xml files to consider")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to directory containing xml files")
    parser.add_argument("--output_dir", type=str, default="output_xmls", help="Path to directory in which to put filtered xml files")
    parser.add_argument("--output_list", type=str, default="output_list.txt", help="Path to output file with list of filtered xmls")

    parser.add_argument("--category", type=str, default="car", help="Name of class to select")

    args = parser.parse_args()

    assert os.path.isdir(args.source_dir), "Source dir does not exist!"
    assert os.path.isfile(args.source_list), "Source list file does not exist!"

    orig_xml_list = get_list_xml_files(args.source_list, args.source_dir)

    counter = 0
    with open(args.output_list, "w") as lf:
        for mf in orig_xml_list:
            contains_class = check_class(mf, args.category)
            if contains_class:
                new_text = remove_other_classes(mf, args.category)
                output_file = get_output_xml(args.output_dir, mf)
                with open(output_file, "w") as f:
                    f.write(new_text)
                lf.write(output_file.split("/")[-1].split(".")[0] + "\n")
                counter += 1
            else:
                print("File {} does not contain class {}".format(mf, args.category))

    print("Kept {} files".format(counter))
