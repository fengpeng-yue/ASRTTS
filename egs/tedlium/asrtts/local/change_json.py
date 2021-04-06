import argparse
import re
import os
import json
import random
import copy
import logging
parser=argparse.ArgumentParser()

parser.add_argument('train_json', help='train json for all feat')
parser.add_argument('remain_feat', help='choice shot feat file')
parser.add_argument('new_json', help='corresponding json')
args=parser.parse_args()

new_json = {}
with open(args.train_json,"rb") as f:
    train_json = json.load(f)["utts"]
with open(args.remain_feat, 'r') as f:
    remain_feat_list = f.read().splitlines()
utt_id_list = []
for feat in remain_feat_list:
    utt_id = feat.split(" ")[0]
    utt_id_list.append(utt_id)

for key in train_json.keys():
    if not key in utt_id_list:
        new_json[key] = train_json[key]


with open(args.new_json, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
