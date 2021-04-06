#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
import azure.cognitiveservices.speech as speechsdk
import editdistance
import re
import unicodedata
import json
import traceback
import soundfile
import librosa
import numpy as np
import argparse

parser=argparse.ArgumentParser()
#parser.add_argument('nj', help='job number')
parser.add_argument('wav_path', help='Input wav file path')
parser.add_argument('text_path', help='Input txt file path')
parser.add_argument('dist_path', help='Output results path')
args=parser.parse_args()
def load_wav(path,sample_rate=16000):
    return librosa.core.load(path, sr=sample_rate)[0]

def save_wav(wav, path,sample_rate=16000):
    wav_ = wav * 1 / max(0.01, np.max(np.abs(wav)))
    soundfile.write(path, wav_.astype(np.float32), sample_rate)
    return path
def trim_silence_intervals(wav,sample_rate=16000):
    intervals = librosa.effects.split(wav, top_db=50,
                                    frame_length=int(sample_rate / 1000 * 25),
                                    hop_length=int(sample_rate / 1000 * 10))
    wav = np.concatenate([wav[l: r] for l, r in intervals])
    return wav
def trim_wav(wav,sample_rate=16000):
    wav_t, _ = librosa.effects.trim(wav, top_db=50,
                                    frame_length=int(sample_rate / 1000 * 25),
                                    hop_length=int(sample_rate / 1000 * 10))
    return wav_t
speech_config = speechsdk.SpeechConfig(subscription="e8ed6b3323094e02813ca45dc3f2d640", region="westus")
# wav_path="/data/t-fyue/disk3/TED_1/TEDLIUM_release1/test/sph/segments/AimeeMullins_2009P-0001782-0002881.wav"
# tmp_dir="/data/t-fyue/disk3/TED_1/TEDLIUM_release1/test/sph/no_slience"
# wav = load_wav(wav_path)
# wav_trim = trim_wav(wav)
# save_wav(wav_trim, os.path.join(tmp_dir, wav_path.split("/")[-1]))
#audio_input = speechsdk.AudioConfig(filename=os.path.join(tmp_dir, wav_path.split("/")[-1]))
result_list=[]
# result_file = args.dist_path + "/hpy.%s.txt" % args.nj
# label_file = args.dist_path + "/ref.%s.txt" % args.nj
result_file = args.dist_path + "/hpy.txt"
label_file = args.dist_path + "/ref.txt"
ref_f = open(label_file,"w")
with open(args.text_path,'r') as txt_f:
    for utt_context in txt_f.read().splitlines():
        label_txt  = utt_context.split(" ",1)[1].lower() + '('+utt_context.split(" ",1)[0] + ')' + '\n'
        ref_f.write(label_txt)
txt_f.close()
ref_f.close()

tmp_dir=args.dist_path + "/no_slience"
if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)
hpy_f = open(result_file,"w")
i=1
with open(args.wav_path,'r') as wav_f:
    for utt_wav_path in wav_f.read().splitlines():
        wav = load_wav(utt_wav_path.split(" ",1)[1])
        wav_trim = trim_silence_intervals(wav)   
        save_wav(wav_trim, os.path.join(tmp_dir, utt_wav_path.split(" ",1)[1].split("/")[-1]))
        audio_input = speechsdk.AudioConfig(filename=os.path.join(tmp_dir, utt_wav_path.split(" ",1)[1].split("/")[-1]))
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input,
                                                    language="en-US")
        result = speech_recognizer.recognize_once_async().get()
        result = json.loads(result.json)
        hpy_f.write(result['DisplayText'].lower()+'('+utt_wav_path.split(" ",1)[0] + ')' + '\n')
        print("order: %d id: %s deocding: %s" %(i,utt_wav_path.split(" ",1)[1].split("/")[-1],result['DisplayText'].lower()))
        result_list.append(result['DisplayText']+'('+utt_wav_path.split(" ",1)[0] + ')')
        if result['RecognitionStatus'] != 'Success':
            print('Failed: ', k, meta[k])
            print(result)
        i = i + 1
    wav_f.close
hpy_f.close()
