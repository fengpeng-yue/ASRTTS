import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('asr_train_other_500', help='Input librispeech train-other-500 unpaired text file path')
parser.add_argument('tts_train_clean_460', help='Input libritts train-clean-460  paired data Path')
parser.add_argument('output_path', help='Output merge josn path')
args=parser.parse_args()


with open(args.asr_train_other_500, "rb") as f:
    asr_json = json.load(f)["utts"]
with open(args.tts_train_clean_460,"rb") as f:
    tts_json = json.load(f)["utts"]

speaker_dict={}
for item in tts_json.keys():
    uttid = item
    xvector_path = tts_json[item]["input"][1]
    speaker_dict[uttid] = xvector_path

choice_dict=copy.deepcopy(speaker_dict)
for item in asr_json.keys():
    if len(choice_dict)==0:
        choice_dict=copy.deepcopy(speaker_dict)
    choice_utt = random.choice(list(choice_dict.keys()))
    xvector_path = choice_dict[choice_utt]
    utt = choice_utt.split('_')
    choice_spk = utt[0]
    asr_json[item]["utt2spk"] = choice_spk
    asr_json[item]["input"].append(xvector_path)
    del choice_dict[choice_utt]


with open(args.output_path, "wb") as f:
        f.write(
            json.dumps(
                {"utts": asr_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
