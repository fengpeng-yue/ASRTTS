import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('speaker_feat', help='Generated speaker')
parser.add_argument('unpaired_json', help='Unpaired josn file')
parser.add_argument('output_path', help='Output josn path')
args=parser.parse_args()
random.seed(1)
speaker_list = []
with open(args.speaker_feat, "r") as f:
    speaker_all=f.read().splitlines()
    #print(speaker_all)
    for speaker in speaker_all:
        speaker_list.append(speaker.split(' '))
with open(args.unpaired_json,"rb") as f:
    unpaired_json = json.load(f)["utts"]
i=0
new_json={}
#print(len(speaker_list))
copy_speaker = copy.deepcopy(speaker_list)
for item in unpaired_json.keys():
    choice_spk = random.choice(copy_speaker)
    speaker = choice_spk[0].split('-')[0]
    embed_path = choice_spk[1]
    new_item = item + "_" +speaker
    # print(item)  
    #print(choice_spk)
    new_json[new_item] = unpaired_json[item]
    new_json[new_item]['input'][0]['feat'] = embed_path
    #new_json[new_item]['utt2spk'] = speaker
    copy_speaker.remove(choice_spk)
    if len(copy_speaker) == 0:
        copy_speaker = copy.deepcopy(speaker_list)

#print(len(speaker_list))
    # if i>80:
    #     break
    # i = i+1
with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
