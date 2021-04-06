. ./path.sh
. ./cmd.sh

stage=0
stop_stage=100
datadir=/data1/fengpeng/data/TEDLIUM_release1
result_prefix=./
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: feature extraction"
    fbankdir=$result_prefix/fbank
    cmvn_path=../../librispeech/asr1/data/train_clean_460/cmvn.ark
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
    # utils/copy_data_dir.sh $result_prefix/data/train $result_prefix/data/train_org
    # remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 5 $result_prefix/data/train_org $result_prefix/data/train

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/train/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/train ${feat_tr_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/test/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/test ${feat_cv_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            $result_prefix/data/dev/feats.scp $cmvn_path  $result_prefix/data/exp/dump_feats/test ${feat_dt_dir}
fi