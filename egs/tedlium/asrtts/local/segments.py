import wave
import argparse
import re
import os
import json
import random
import copy
parser=argparse.ArgumentParser()

parser.add_argument('wav_path', help='Input wav file path')
parser.add_argument('segments', help='Input segments Path')
#parser.add_argument('output_path', help='Output merge josn path')
args=parser.parse_args()
#segments=[]
last_record=None
with open(args.segments, "r") as segments_f:
    date = segments_f.read().splitlines()
    for record in date:
        #print(record)
        segments = record.split(" ")
        if last_record is None or last_record != segments[1]:
            last_record = segments[1]
            os.system('sox %s/%s.sph %s/%s.wav' % (args.wav_path,segments[1],args.wav_path,segments[1]))
            wf = wave.open(os.path.join(args.wav_path,segments[1]+".wav"), 'rb') # 二进制读文件

            wav_chn = wf.getnchannels()  		# 获取通道数
            wav_sampwidth = wf.getsampwidth()   # 比特宽度 每一帧的字节数 2B--2个字节-2*8=16bit(bit二进制位) 采样位数
            wav_samprate = wf.getframerate()	# 帧率  每秒有多少帧 
            wav_framenumber = wf.getnframes()	# 当前音频的总帧数

            wav_data = wf.readframes(wav_framenumber) # 按帧读音频，返回二进制字符串数据 读取音频所有数据
            wf.close()
        
        start = int(float(segments[2])*1000)
        #print(start)
        end = int(float(segments[3])*1000)
        #print(end)
        buff = wav_data[start * 16 * 2: end * 16 * 2]  # 字符串索引切割
        #write_wav('xxx' + '.wav'), buff)
        path = args.wav_path + "/segments" +"/" + segments[0] + ".wav"
        wav_f = wave.open(path, 'wb')
        wav_f.setnchannels(wav_chn)
        wav_f.setsampwidth(wav_sampwidth)
        wav_f.setframerate(wav_samprate)
        wav_f.writeframes(buff) # buff变量值
        wav_f.close()
        #break


