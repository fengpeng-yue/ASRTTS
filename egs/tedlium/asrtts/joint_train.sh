. ./path.sh
. ./cmd.sh


# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=8      # number of gpus ("0" uses cpu, otherwise use gpu)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nnodes=1
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
train_asr_config=conf/train_asr.yaml
train_asrtts_config=conf/train_asrtts.yaml
train_tts_config=conf/train_pytorch_tacotron2+spkemb.yaml


# training related
asr_opt=adadelta
tts_opt=adam



asr_batch_size=20
tts_batch_size=20
unpaired_batch_size=6

tts_grad_clip=1

parallel_mode=ddp  #choice ["ddp","dp"]

flag=only_update_asr_stage1

shuffle_spk=false
preprocess_config=conf/specaug.yaml

# for the three-stage training
update_asr=true
update_tts=false
update_tts2asr=true
filter_data=true
filter_thre=0.58 
unpaired_aug=true 
tts_loss_weight=0.005


mix_precision=true
use_inference=true
asrtts_train=true



result_prefix=$(pwd)
. utils/parse_options.sh || exit 1;


asr_dir=../../librispeech/asr1
asrexpdir=${asr_dir}/exp/asr_baseline
asrexpdir=exp/asrtts_joint_traing_only_update_asr_bpe

asr_paired_data=${asr_dir}/dump/train_clean_460/data.json
asr_dev_json=dump/dev/data.json
asr_model_conf=$asrexpdir/results/model.json
asr_model=$asrexpdir/results/model.acc.best
asr_dict=${asr_dir}/data/lang_char/train_clean_460_units.txt



tts_dir=../../libritts/tts1
ttsexpdir=${tts_dir}/exp/train_clean_460_pytorch_baseline



tts_paired_data=${tts_dir}/dump/train_clean_460/data_phone_tts.json
tts_dev_json=${tts_dir}/dump/dev_clean/data_phone_tts.json


tts_model=$ttsexpdir/results/model.loss.best
tts_model_conf=$ttsexpdir/results/model.json


feat_tr_up_dir=$result_prefix/dump_asrtts/train
asrttsexpdir=$result_prefix/exp/asrtts_joint_traing_$flag


unpaired_data=${feat_tr_up_dir}/data_adap.json




tr_json_list="${unpaired_data} ${asr_paired_data} ${tts_paired_data}"


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
        --node_rank 1 \
        --master_addr='10.20.60.140' \
        --master_port 29508 \
        ../../../espnet/bin/asrtts_train.py "
        # train_cmd="python
        # ../../../espnet/bin/asrtts_train.py "
    fi
    ${cuda_cmd} --gpu 1 ${asrttsexpdir}/train.log \
    $train_cmd \
    --use_launch true \
    --world_size 3 \
    --node_rank 1 \
    --master_addr "10.20.10.26" \
    --master_port 22335 \
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
    --tts-grad-clip ${tts_grad_clip} \
    --mix-precision ${mix_precision} \
    --filter-data $filter_data \
    --filter-thre $filter_thre \
    --unpaired-aug $unpaired_aug \
    --tts-loss-weight $tts_loss_weight
fi