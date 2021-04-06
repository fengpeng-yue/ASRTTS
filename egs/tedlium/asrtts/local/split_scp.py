#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('nj', type=int,help='job number')
parser.add_argument('scp_path', help='Input scp file path')
parser.add_argument('output_dir',help='output directory')

args=parser.parse_args()

scp_file = open(args.scp_path,"r").read().splitlines()
total_num = len(scp_file)
print(total_num)

each_num = int(total_num/args.nj)
j=0
for i in range(1,args.nj+1):
    each_scp_file = open(args.output_dir+"/feats.%s.scp"%i,"w")
    while(j>=each_num*(i-1) and j <each_num*i):
        if j >=total_num:
            each_scp_file.close()
            break
        each_scp_file.write(scp_file[j] + "\n")
        j += 1
while True:
    if j < total_num:
        each_scp_file.write(scp_file[j]+"\n")
        j += 1
    else:
        break
each_scp_file.close()
