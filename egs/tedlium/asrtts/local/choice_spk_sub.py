import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('training_set_ft', help='Training set feat path')
parser.add_argument('test_set_ft', help='Test set feat path')
parser.add_argument('training_sub', help='Training subset feat path')
args=parser.parse_args()

train_spk = []
test_spk = []

with open(args.test_set_ft,'r') as f:
    test_set = f.read().splitlines()
    for item in test_set:
        speaker = item.split(' ')[0].split('_')[0]
        test_spk.append(speaker)

test_spk = set(test_spk)
print(test_spk)
with open(args.training_sub,'w') as sub_f:
    with open(args.training_set_ft,'r') as f:
        training_set = f.read().splitlines()
        for item in training_set:
            speaker = item.split(' ')[0].split('_')[0]
            if speaker in test_spk:
                item = item + '\n'
                sub_f.write(item)
        f.close()
    sub_f.close()
    
        


    
