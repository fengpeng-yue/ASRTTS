#!/bin/bash


. ./path.sh
. ./cmd.sh


result_prefix=$(pwd)
# general configuration
backend=pytorch
stage=0       # start from 0
stop_stage=100
ngpu=4        # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=$result_prefix/dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
#resume=       # Resume the training from snapshot
preprocess_config=conf/specaug.yaml

# feature configuration
do_delta=false
train_asr_config=conf/train_asr.yaml
lm_config=conf/lm.yaml
decode_asr_config=conf/decode_asr.yaml

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80    # number of mel basis(if you haven't enough memory, you can set it to 80)
n_fft=800    # number of fft points
n_shift=160   # number of shift points
win_length="" # window length



# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
datadir=


# bpemode (unigram or bpe)
nbpe=5000
bpemode=bpe
use_bpe=true

# training related
tag=baseline
opt=adadelta
nnodes=1
parallel_mode=ddp

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_460
train_dev=dev
recog_set="test_clean"


nj=20
dev_set=$train_dev
eval_set=test_clean
fbankdir=$result_prefix/fbank
feat_tr_dir=${dumpdir}/${train_set}; 
feat_dt_dir=${dumpdir}/${dev_set};
feat_ev_dir=${dumpdir}/${eval_set};
dict=$result_prefix/data/lang_char/${train_set}_units.txt
nlsyms=$result_prefix/data/lang_char/non_lang_syms.txt
bpemodel=$result_prefix/data/lang_char/${train_set}_${bpemode}${nbpe}
. utils/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature extraction"
    for x in dev_clean test_clean train_clean_100 train_clean_360; do
        if [ ! -s data/${x}/feats.scp ]; then
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
        fi
    done
    utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360
    utils/combine_data.sh $result_prefix/data/${dev_set}_org $result_prefix/data/dev_clean

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 $result_prefix/data/${dev_set}_org $result_prefix/data/${dev_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 $result_prefix/data/${train_set}_org $result_prefix/data/${train_set}
    
    # It is better to use same CMVN for ASR and TTS training.
    #compute-cmvn-stats scp:$result_prefix/data/${train_set}/feats.scp $result_prefix/data/${train_set}/cmvn.ark
    cat $result_prefix/data/${train_set}/feats.scp ../../libritts/tts1/data/train_clean_460/feats.scp > $result_prefix/data/${train_set}/feats_all.scp
    compute-cmvn-stats scp:$result_prefix/data/${train_set}/feats_all.scp $result_prefix/data/${train_set}/cmvn_all.ark

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/${train_set}/feats.scp $result_prefix/data/${train_set}/cmvn_all.ark  $result_prefix/data/exp/dump_feats/train ${feat_tr_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $result_prefix/data/${dev_set}/feats.scp $result_prefix/data/${train_set}/cmvn_all.ark  $result_prefix/data/exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        $result_prefix/data/${eval_set}/feats.scp  $result_prefix/data/${train_set}/cmvn_all.ark $result_prefix/exp/dump_feats/eval ${feat_ev_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: JSON preparation"
    echo "dictionary: ${dict}"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p $result_prefix/data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    if [ $use_bpe == 'true' ]; then
        cut -f 2- -d" " $result_prefix/data/${train_set}/text > $result_prefix/data/lang_char/input.txt
        spm_train --input=$result_prefix/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
        spm_encode --model=${bpemodel}.model --output_format=piece < $result_prefix/data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    else
        echo "make a non-linguistic symbol list"
        #cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
        #cat ${nlsyms}
        text2token.py -s 1 -n 1 $result_prefix/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    fi
    wc -l ${dict}
    # make json labels
    if [ ! -s ${feat_ev_dir}/data.json ]; then
        data2json.sh --feat ${feat_tr_dir}/feats.scp \
            $result_prefix/data/${train_set} ${dict} > ${feat_tr_dir}/data.json
        data2json.sh --feat ${feat_dt_dir}/feats.scp \
            $result_prefix/data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
        data2json.sh --feat ${feat_ev_dir}/feats.scp \
            $result_prefix/data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
    fi
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: ASR training"
    
    expdir=$result_prefix/exp/asr_${tag}
    expname=asr_${tag}
    resume=
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
    if [ $parallel_mode == 'ddp' ]; then
        echo $parallel_mode
        train_cmd="python -m torch.distributed.launch \
        --nproc_per_node=$nproc_per_node \
        --nnodes ${nnodes} \
        --master_port 29507  \
        ../../../espnet/bin/asr_train.py"
    else
        train_cmd="asr_train.py"
    fi
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    $train_cmd \
    --ngpu ${ngpu} \
    --config ${train_asr_config} \
    --preprocess-conf ${preprocess_config} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir $result_prefix/tensorboard/${expname} \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --parallel-mode $parallel_mode \
    --opt ${opt} \
    --train-json ${feat_tr_dir}/data.json \
    --valid-json ${feat_dt_dir}/data.json
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 3: ASR decoding"
    nj=6
    ngpu=0
    pids=() # initialize pids
    for rtask in ${eval_set}; do
    (
        decode_dir=decode_${rtask}
        feat_recog_dir=${dumpdir}/${rtask}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        #### use CPU for decoding
        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --config $decode_asr_config \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            

        if [ $use_bpe == 'true' ]; then
            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        else
            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
