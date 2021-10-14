#!/bin/bash


. ./path.sh
. ./cmd.sh

stage=0
stop_stage=100
nj=20
datadir=
asr_dir=../../librispeech/asr1
tts_dir=../../libritts/tts1
result_prefix=$(pwd)

dumpdir=$result_prefix/dump
feat_tr_dir=${dumpdir}/train; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/dev; mkdir -p ${feat_dt_dir}
feat_cv_dir=${dumpdir}/test; mkdir -p ${feat_cv_dir}

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=800    # number of fft points
n_shift=160   # number of shift points
win_length="" # window length

trans_type="phn"


. utils/parse_options.sh || exit 1;
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/prepare_data.sh ${datadir} $result_prefix/data
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: speech segments"
    for x in dev test train; do
        mkdir -p ${datadir}/$x/sph/segments
        python local/segments.py ${datadir}/$x/sph $result_prefix/data/$x/tmp/segments
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature extraction"
    fbankdir=$result_prefix/fbank
    cmvn_path=$asr_dir/data/train_clean_460/cmvn_all.ark
    asr_dict=$asr_dir/data/lang_char/train_clean_460_units.txt
    for x in dev test train; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            $result_prefix/data/$x \
            $result_prefix/exp/make_fbank/$x \
            ${fbankdir}
    done
    utils/copy_data_dir.sh $result_prefix/data/train $result_prefix/data/train_org
    #remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 5 $result_prefix/data/train_org $result_prefix/data/train

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/train/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/train ${feat_tr_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/test/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/test ${feat_cv_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/dev/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/test ${feat_dt_dir}

    data2json.sh --feat ${feat_tr_dir}/feats.scp  \
        $result_prefix/data/train ${asr_dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_cv_dir}/feats.scp  \
        $result_prefix/data/test ${asr_dict} > ${feat_cv_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp  \
        $result_prefix/data/dev ${asr_dict} > ${feat_dt_dir}/data.json
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    tts_dict=$tts_dir/data/lang_1phn/train_clean_460_units.txt
    for x in dev test train; do
        local/clean_text.py $result_prefix/data/${x}/text_punc $trans_type > $result_prefix/data/${x}/text_phone
    done
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/dev/text_phone\
            $result_prefix/data/dev ${tts_dict} > ${feat_dt_dir}/data_phone.json
    data2json.sh --feat ${feat_cv_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/test/text_phone\
            $result_prefix/data/test ${tts_dict} > ${feat_cv_dir}/data_phone.json
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} --text $result_prefix/data/train/text_phone\
            $result_prefix/data/train ${tts_dict} > ${feat_tr_dir}/data_phone.json
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    nj=11
    # only for test set
    for name in test; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # # # Check pretrained model existence
    nnet_dir=$result_prefix/exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a $result_prefix/exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # # Extract x-vector
    for name in test; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Josn for joint training "
    python local/generate_unpaired.py $tts_dir/exp/xvector_nnet_1a/xvectors_train_clean_460/xvector.scp \
                                        $result_prefix/dump/train/data_phone.json \
                                        $result_prefix/dump/train/data.json \
                                        0 \
                                        $result_prefix/dump/train/data_adap.json 
fi