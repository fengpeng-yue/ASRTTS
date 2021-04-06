#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
# . ./path_new_docker.sh
. ./cmd.sh || exit 1;

maxframes=2000
result_prefix=/data1/fengpeng/espnet-2021-4-6/egs/libritts/tts1
ngpu=1
# general configuration
backend=pytorch
stage=5
stop_stage=100
#ngpu=1       # number of gpu in training
nj=20        # number of parallel jobs
dumpdir=$result_prefix/dump # directory to dump full features
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=800    # number of fft points
n_shift=160   # number of shift points
win_length="" # window length




# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/data1/fengpeng/data
#datadir=/data1/baibing/datasets

# base url for downloads.
data_url=www.openslr.org/resources/60


#optimizer
opt=lr-decay
initial_lr=1e-3
final_lr=1e-5
decay_rate=0.5
decay_steps=12500
warmup_steps=50000

batch_size=20


#distributed training
nnodes=1
parallel_mode=ddp  #choice ["ddp","dp"]
# exp tag
tag=baseline # tag for managing experiments. # tag for managing experiments.

#gudied loss
use_guided_attn_loss=true



. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_460
dev_set=dev_clean
eval_set=test_clean

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}



trans_type="phn"
dict=$result_prefix/data/lang_1${trans_type}/${train_set}_units.txt
#nnet_dir=$result_prefix/exp/xvector_nnet_1a
nnet_dir=$result_prefix/exp/spk_embedding


# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data preparation"
#     for part in dev-clean test-clean train-clean-100 train-clean-360; do
#         # use underscore-separated names in data directories.
#         local/data_prep.sh ${datadir}/LibriTTS/${part} data/${part//-/_}
#     done
# fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=$result_prefix/fbank
    for x in dev_clean; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            $result_prefix/data/${x} \
            $result_prefix/exp/make_fbank/${x} \
            ${fbankdir}
    done

    utils/combine_data.sh $result_prefix/data/${train_set}_org $result_prefix/data/train_clean_100 $result_prefix/data/train_clean_360
    utils/combine_data.sh $result_prefix/data/${dev_set}_org $result_prefix/data/dev_clean

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    
    remove_longshortdata.sh --maxframes $maxframes --maxchars 400 $result_prefix/data/${train_set}_org $result_prefix/data/${train_set}
    remove_longshortdata.sh --maxframes $maxframes --maxchars 400 $result_prefix/data/${dev_set}_org $result_prefix/data/${dev_set}

    #compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:$result_prefix/data/${train_set}/feats.scp $result_prefix/data/${train_set}/cmvn.ark
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    local/clean_text.py $result_prefix/data/${train_set}/text $trans_type > $result_prefix/data/${train_set}/text_phone
    local/clean_text.py $result_prefix/data/${dev_set}/text $trans_type > $result_prefix/data/${dev_set}/text_phone
    local/clean_text.py $result_prefix/data/${eval_set}/text $trans_type > $result_prefix/data/${eval_set}/text_phone

    mkdir -p $result_prefix/data/lang_1${trans_type}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 --trans_type ${trans_type} $result_prefix/data/${train_set}/text_phone | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $result_prefix/data/${train_set}/feats.scp $result_prefix/data/${train_set}/cmvn.ark $result_prefix/exp/dump_feats/train ${feat_tr_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $result_prefix/data/${dev_set}/feats.scp $result_prefix/data/${train_set}/cmvn.ark $result_prefix/exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $result_prefix/data/${eval_set}/feats.scp $result_prefix/data/${train_set}/cmvn.ark $result_prefix/exp/dump_feats/eval ${feat_ev_dir}
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/${dev_set}/text_phone\
            $result_prefix/data/${dev_set} ${dict} > ${feat_dt_dir}/data_phone.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/${eval_set}/text_phone\
            $result_prefix/data/${eval_set} ${dict} > ${feat_ev_dir}/data_phone.json
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/${train_set}/text_phone\
            $result_prefix/data/${train_set} ${dict} > ${feat_tr_dir}/data_phone.json
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    # mfccdir=mfcc
    # vaddir=mfcc
    # for name in ${train_set} ${dev_set} ${eval_set}; do
    #     utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
    #     utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
    #     steps/make_mfcc.sh \
    #         --write-utt2num-frames true \
    #         --mfcc-config conf/mfcc.conf \
    #         --nj ${nj} --cmd "$train_cmd" \
    #         data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
    #     utils/fix_data_dir.sh data/${name}_mfcc_16k
    #     sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
    #         data/${name}_mfcc_16k exp/make_vad ${vaddir}
    #     utils/fix_data_dir.sh data/${name}_mfcc_16k
    # done

    # # # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    # if [ ! -e ${nnet_dir} ]; then
    #     echo "X-vector model does not exist. Download pre-trained model."
    #     wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
    #     tar xvf 0008_sitw_v2_1a.tar.gz
    #     mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
    #     rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    # fi
    # # Extract x-vector
    for name in ${dev_set} ${eval_set} ${train_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    for name in ${train_set} ${dev_set} ${eval_set}; do
        cp ${dumpdir}/${name}/data_phone.json ${dumpdir}/${name}/data_phone_tts.json
        #if [ $name == ${train_paired_set} ]; then fname=${train_set}; else fname=$name; fi
        local/update_json.sh ${dumpdir}/${name}/data_phone_tts.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
fi



if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=$result_prefix/exp/${expname}
mkdir -p ${expdir}
for ((i=60;i>=0;i=i-1)); do
    resume=${expdir}/results/snapshot.ep.$i
    echo $resume
    if [ -f $resume ]; then
        break;
    else
        resume=
    fi
done
echo $resume
nproc_per_node=$ngpu
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data_phone_tts.json
    dt_json=${feat_dt_dir}/data_phone_tts.json
    #${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    if [ $parallel_mode == 'ddp' ]; then
        train_cmd="python -m torch.distributed.launch \
        --nproc_per_node=$nproc_per_node \
        --nnodes ${nnodes} \
        --master_port 29505  \
        ../../../espnet/bin/tts_train.py"
    else
        train_cmd="tts_train.py"
    fi
    ${cuda_cmd} --gpu 1 ${expdir}/train.log \
    $train_cmd \
    --backend ${backend} \
    --use-guided-attn-loss ${use_guided_attn_loss} \
    --opt ${opt} \
    --initial-lr ${initial_lr} \
    --final-lr ${final_lr} \
    --decay-rate ${decay_rate} \
    --decay-steps ${decay_steps} \
    --warmup-steps ${warmup_steps} \
    --ngpu ${ngpu} \
    --outdir ${expdir}/results \
    --tensorboard-dir $result_prefix/tensorboard/${expname} \
    --verbose ${verbose} \
    --seed ${seed} \
    --resume ${resume} \
    --train-json ${tr_json} \
    --valid-json ${dt_json} \
    --num-iter-processes 0 \
    --parallel-mode ${parallel_mode} \
    --batch-size ${batch_size} \
    --config ${train_config} 
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data_phone_tts_new_spk.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data_phone_tts_new_spk.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data_phone_tts_new_spk.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    for name in ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true $result_prefix/data/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
