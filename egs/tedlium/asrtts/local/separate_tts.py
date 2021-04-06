#!/usr/bin/env python3
# encoding: utf-8
import torch
import numpy as np
import os
import logging
import argparse
from collections import OrderedDict
parser=argparse.ArgumentParser()

parser.add_argument('input_snapshot_path', help='Input snapshot.*.ep')
parser.add_argument('output_ttsmodel_path', help='Output separated tts/asr model')
parser.add_argument('mode', help='mode,asr or tts')

args=parser.parse_args()
if args.mode == "asr":
    print("Separate ASR model")
    states = torch.load(args.input_snapshot_path, map_location=torch.device("cpu"))["asr_model"]
elif args.mode == "tts":
    print("Separate TTS model")
    states = torch.load(args.input_snapshot_path, map_location=torch.device("cpu"))["tts_model"]
new_states=OrderedDict()
print(type(states))
for name, param in states.items():
    name = name.replace("model.", "")
    new_states[name] = param
torch.save(new_states, args.output_ttsmodel_path)
# print(states)
# np.savez_compressed(args.output_ttsmodel_path, **states)
# os.rename("{}.npz".format(args.output_ttsmodel_path), args.output_ttsmodel_path)
# states = torch.load(args.input_snapshot_path, map_location=torch.device("cpu"))
# print(type(states))