import wave
import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('i_wav_path', help='Input wav file path')
parser.add_argument('o_wav_path', help='output wav file Path')

args=parser.parse_args()
last_record=None
with open(args.i_wav_path, "r") as i_wav_path_f:
    date = i_wav_path_f.read().splitlines()
    for record in date:
        #print(record)
        wav_path = record.split(" ")[1]
        wf = wave.open(wav_path, 'rb') # 二进制读文件

        wav_chn = wf.getnchannels()  		# 获取通道数
        wav_sampwidth = wf.getsampwidth()   # 比特宽度 每一帧的字节数 2B--2个字节-2*8=16bit(bit二进制位) 采样位数
        wav_samprate = wf.getframerate()	# 帧率  每秒有多少帧 
        wav_framenumber = wf.getnframes()	# 当前音频的总帧数

        wav_data = wf.readframes(wav_framenumber) # 按帧读音频，返回二进制字符串数据 读取音频所有数据
        wf.close()
        
        length = len(wav_data)
        print(length)
        start = 1*100
        #print(start)
        end = length-200
        #print(end)
        buff = wav_data[start * 16 * 2: end * 16 * 2]  # 字符串索引切割
        #write_wav('xxx' + '.wav'), buff)
        wav_name = wav_path.split("/")[-1]
        path = args.o_wav_path +"/"+ wav_name
        wav_f = wave.open(path, 'wb')
        wav_f.setnchannels(wav_chn)
        wav_f.setsampwidth(wav_sampwidth)
        wav_f.setframerate(wav_samprate)
        wav_f.writeframes(buff) # buff变量值
        wav_f.close()