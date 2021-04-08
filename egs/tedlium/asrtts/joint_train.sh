#!/bin/bash


. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stop_stage=100
ngpu=8        # number of gpus 
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
train_asr_config=conf/train_asr.yaml
train_asrtts_config=conf/train_asrtts.yaml
train_tts_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_tts_config=conf/decode_tts.yaml
decode_asr_config=conf/decode_asr.yaml

# training related
asr_opt=adadelta
tts_opt=adam

asr_batch_size=20
tts_batch_size=60
unpaired_batch_size=60

parallel_mode=ddp  #choice ["ddp","dp"]
flag=
shuffle_spk=false
preprocess_config=conf/specaug.yaml
update_asr=true
update_tts=true
update_tts2asr=true

mix_precision=true
use_inference=true
asrtts_train=true

method="domain" #[domain or speaker]
shot_num=5      #[1 or 5]
nnodes=1
result_prefix=$(pwd)



asr_dir=../../librispeech/asr1
asrexpdir=${asr_dir}/exp/asr_baseline
asr_paired_data=${asr_dir}/dump/train_clean_460/data.json
asr_dev_json=${asr_dir}/dump/dev/data.json
asr_model_conf=$asrexpdir/results/model.json
asr_model=$asrexpdir/results/model.acc.best
asr_dict=${asr_dir}/data/lang_char/train_clean_460_units.txt


tts_dir=../../libritts/tts1
ttsexpdir=${tts_dir}/exp/train_clean_460_pytorch_baseline
tts_paired_data=${tts_dir}/dump/train_clean_460/data_phone_tts.json
tts_dev_json=${tts_dir}/dump/dev_clean/data_phone_tts.json
tts_model=$ttsexpdir/results/model.loss.best
tts_model_conf=$ttsexpdir/results/model.json


feat_tr_up_dir=$result_prefix/dump/train
asrttsexpdir=$result_prefix/exp/asrtts_joint_traing_$flag


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
if [ $method == "domain" ]; then
    unpaired_data=${feat_tr_up_dir}/data_adap.json
else
    if [ ${shot_num} == 1 ];then
        unpaired_data=${feat_tr_up_dir}/data_one_shot_adap.json
    elif [ ${shot_num} == 5 ];then
        unpaired_data=${feat_tr_up_dir}/data_five_shot_adap.json
    fi
fi


tr_json_list="${unpaired_data} ${asr_paired_data} ${tts_paired_data} "


for ((i=60;i>=0;i=i-1)); do
    resume=${asrttsexpdir}/results/snapshot.ep.$i
    echo $resume
    if [ -f $resume ]; then
        break;
    else
        resume=
    fi
done
echo $resume
nproc_per_node=$ngpu

if [ $asrtts_train == 'true' ]; then
    if [ $parallel_mode == 'dp' ];then
        train_cmd="asrtts_train.py "
    else
        train_cmd="python -m torch.distributed.launch \
        --nproc_per_node=$nproc_per_node \
        --nnodes $nnodes \
        --master_port 29507 \
        ../../../espnet/bin/asrtts_train.py "
    fi
    ${cuda_cmd} --gpu 1 ${asrttsexpdir}/train.log \
    $train_cmd \
    --config ${train_asrtts_config} \
    --preprocess-conf ${preprocess_config} \
    --ngpu $ngpu \
    --backend ${backend} \
    --outdir ${asrttsexpdir}/results \
    --tensorboard-dir $result_prefix/tensorboard/asrtts_$flag \
    --debugmode ${debugmode} \
    --dict ${asr_dict} \
    --debugdir ${asrttsexpdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${tr_json_list} \
    --asr-valid-json  ${asr_dev_json} \
    --tts-valid-json  ${tts_dev_json} \
    --asr-opt ${asr_opt} \
    --tts-opt ${tts_opt} \
    --asr-model-conf $asr_model_conf \
    --asr-model $asr_model \
    --tts-model-conf $tts_model_conf \
    --tts-model $tts_model \
    --parallel-mode ${parallel_mode} \
    --shuffle-spk ${shuffle_spk} \
    --asr-batch-size ${asr_batch_size} \
    --tts-batch-size ${tts_batch_size} \
    --unpaired-batch-size ${unpaired_batch_size} \
    --update-asr ${update_asr} \
    --update-tts ${update_tts} \
    --update-tts2asr ${update_tts2asr} \
    --mix-precision ${mix_precision} 
fi