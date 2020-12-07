import argparse
import torch


parser = argparse.ArgumentParser("Script to convert pytorch 1.6 models in pytorch 1.4-compatible models. Must be executed with pytorch v1.6+")

parser.add_argument("--input", required=True, type=str, help="Input pytorch checkpoint (pytorch 1.6)")
parser.add_argument("--output", required=True, type=str, help="Output pytorch checkpoint (pytorch 1.4)")

args = parser.parse_args()

a = torch.load(args.input)

torch.save(a, args.output, _use_new_zipfile_serialization=False)
