#!/usr/bin/env python3
import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('speaker_feat', help='Generated speaker')
parser.add_argument('unpaired_phone', help='Unpaired phone josn file')
parser.add_argument('unpaired_char', help='Unpaired phone josn file')
parser.add_argument('shot_num',default=0,type=int,
                    help="0: use all speaker embeddings \
                          1: use one speaker embedding for each speaker\
                          5: use five speaker embeddings for each speaker")
parser.add_argument('output_path', help='Output josn path')
args=parser.parse_args()
random.seed(1)

speaker_dict = {}
"""
{
speaker1:[{utt_id1:speaker_embedding_path},{utt_id2:speaker_embedding_path}...],
speaker2:[{utt_id1:speaker_embedding_path},{utt_id1:speaker_embedding_path}...],
...
}
"""
embedding_list=[]
with open(args.speaker_feat, "r") as f:
    speaker_all=f.read().splitlines()
    for item in speaker_all:
        utt_id = item.split(" ")[0]
        speaker = utt_id.split("_")[0]
        embedding_path = item.split(" ")[1]
        #embedding_list.append(embedding_path)
        if not speaker in speaker_dict:
            speaker_dict[speaker] =[]
            speaker_dict[speaker].append({utt_id:embedding_path})
        else:
            speaker_dict[speaker].append({utt_id:embedding_path})
if args.shot_num !=0:
    for speaker,speaker_list in speaker_dict.items():
        for i in range(args.shot_num):
            pair = random.choice(speaker_list)
            embedding_path = list(pair.values())[0]
            # print(embedding_path)
            speaker_list.remove(pair)
            embedding_list.append(embedding_path)
else:
    for speaker,speaker_list in speaker_dict.items():
        for item in speaker_list:
            embedding_path = list(item.values())[0]
            # print(embedding_path)
            embedding_list.append(embedding_path)

with open(args.unpaired_phone,"rb") as f:
    unpaired_phone = json.load(f)["utts"]

with open(args.unpaired_char,"rb") as f:
    unpaired_char = json.load(f)["utts"]

new_json={}

copy_list = copy.deepcopy(embedding_list)
print(len(copy_list))

for item in unpaired_char.keys():
    choice_spk = random.choice(copy_list)
    unpaired_char[item]["input"][0]["feat"] = choice_spk
    unpaired_char[item]["input"][0]["shape"] = [512]
    unpaired_char[item]["output"].append(unpaired_phone[item]["output"][0])
    unpaired_char[item]["output"][1]["name"] = "target2"
    new_json[item] = unpaired_char[item]

    copy_list.remove(choice_spk)
    if len(copy_list) == 0:
        copy_list = copy.deepcopy(embedding_list)
with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
