import argparse
import re
import os
import json

parser=argparse.ArgumentParser()

parser.add_argument('input_char_path', help='Input char josn Path')
parser.add_argument('input_phone_path', help='Input phone josn Path')
parser.add_argument('output_path', help='Output merge josn path')
args=parser.parse_args()
print("adfasdfasdf")
with open(args.input_char_path, "rb") as f:
    char_json = json.load(f)["utts"]
print("adfasdfasdf")
with open(args.input_phone_path, "rb") as f:
    phone_json = json.load(f)["utts"]
print("adfasdfasdf")
for key in char_json.keys():
    phone_json[key]['output'][0]['name'] =  "target2"
    char_json[key]['output'].append(phone_json[key]['output'][0])
print("adfasdfasdf")
with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": char_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )

    



