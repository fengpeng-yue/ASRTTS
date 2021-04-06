import argparse
import re
import os
import json

parser=argparse.ArgumentParser()

parser.add_argument('input_path', help='Input File Path')
parser.add_argument('output_path', help='train output file path')
args=parser.parse_args()

with open(args.input_path, "rb") as f:
    train_json = json.load(f)["utts"]

for key in train_json.keys():
    train_json[key]['input'].pop(0)
with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": train_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )