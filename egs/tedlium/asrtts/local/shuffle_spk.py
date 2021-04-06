import argparse
import re
import os
import json
import random
import copy
import logging
parser=argparse.ArgumentParser()

parser.add_argument('test_clean', help='Test josn for shuffle spk')
parser.add_argument('output_path', help='Output josn path')
args=parser.parse_args()

with open(args.test_clean,"rb") as f:
    tts_json = json.load(f)["utts"]
changed_json = tts_json
speaker_dict={}
for item in tts_json.keys():
    uttid = item
    xvector_path = tts_json[item]["input"][1]["feat"]
    speaker_dict[uttid] = xvector_path

choice_dict=copy.deepcopy(speaker_dict)
#print(choice_dict)
for item in changed_json.keys():
    if len(choice_dict)==0:
        choice_dict=copy.deepcopy(speaker_dict)
    choice_utt = random.choice(list(choice_dict.keys()))
    print(choice_utt)
    xvector_path = choice_dict[choice_utt]
    utt = choice_utt.split('-')
    choice_spk = utt[0]
    changed_json[item]["utt2spk"] = choice_spk
    print(changed_json[item]["utt2spk"])
    changed_json[item]["input"][1]["feat"] = xvector_path
    print(changed_json[item]["input"][1]["feat"])
    del choice_dict[choice_utt]


with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": changed_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
