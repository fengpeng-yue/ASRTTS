#. ./path.sh
. ./path_new_docker.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
stop_stage=100
#ngpu=1        # number of gpus ("0" uses cpu, otherwise use gpu)
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
lm_config=conf/lm.yaml
decode_asr_config=conf/decode_asr.yaml


# decoding parameter
lm_weight=0.0
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.0
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'



# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
use_bpe=false

# training related
asrtts_train=true
asrtts_decode=false

asr2tts_policy=straight_through # choice ["policy_gradient","straight_through"]
use_rnnlm=false
asr_opt=adadelta
tts_opt=adam

train_mode=1
asr_batch_size=1
tts_batch_size=1
unpaired_batch_size=60
test=false
use_aug=true
parallel_mode=ddp  #choice ["ddp","dp"]
flag=update-save-spec
shuffle_spk=false
preprocess_config=conf/specaug.yaml
update_asr=true
update_tts=true
update_tts2asr=true
update_spk=false
use_spk=false
warmup_iter=-1
mix_precision=true
use_inference=true
unpaired_aug=false
spk_loss=dist   # distance
shot_num=5
nnodes=1
iterative_training=false
#preprocess_config=
gpu_server=8
. utils/parse_options.sh || exit 1;

#set experiment name


if [ $(hostname) == stcgpu-17 ]; then
    ttsdata_prefix=/data/t-fyue/disk2/results/libritts_results
    unpaireddata_prefix=/data/t-fyue/disk2/results/ted_results
    result_prefix=/data/t-fyue/disk2/results/asrtts_results
    if [ $spk_loss == "dist" ]; then
        spkexpdir=$result_prefix/exp/speaker_model/spk_ver
        spk_model=$spkexpdir/model000000029.model
    else
        spkexpdir=$result_prefix/exp/speaker_model/no_max_len
        spk_model=$spkexpdir/model000000155.model
    fi
    utt2spk=/data/t-fyue/disk2/results/libritts_results/data/train_clean_460/spk_recog_utt2spk
    if [ $use_aug == "true" ]; then
        #asrexpdir=$result_prefix/exp/asr_baseline_agu_true_newencoder_decode-2_larger_epoch25_distributed_mask_zero_sparse_image_warp
        asrexpdir=$result_prefix/exp/asr_baseline_agu_true_test_aug_false_newencoder_larger_epoch45_libritts_cmvn
    else
        asrexpdir=$result_prefix/exp/asr_baseline
    fi
    #asrexpdir=~/disk2/result/test_model/asr_baseline_agu_false_larger_epoch25
    #asrexpdir=~/disk2/result/test_model/asr_baseline_agu_true_newencoder_decode-2_larger_epoch25_espnet_new
    echo "traing mode is ${train_mode}"
    if [ ${train_mode} == 1 ]; then
        echo "using phone model"
        #ttsexpdir=~/disk2/result/librispeech_newdata/exp/tts_baseline_phone
        ttsexpdir=$result_prefix/exp/train_clean_460_pytorch_phone-lr-decay_16k_maxframe_2000_guided_reduction_factor1_batch128_gpu2-true
        #ttsexpdir=$result_prefix/exp/train_clean_460_pytorch_phone_lr-decay_16k_maxframe_2000_use-guided_reduction_factor1_batch128_gpu8_ddp_new_spk
        #ttsexpdir=$result_prefix/exp/train_clean_460_pytorch_phone_lr-decay_maxframes_3000
        tag=asr-batch_${asr_batch_size}_tts-batch_${tts_batch_size}_unpaired-batch_${unpaired_batch_size}_unpaired-use-aug_${unpaired_aug}_use-spk_${use_spk}_shuffle-spk_${shuffle_spk}_spk-loss_${spk_loss}_flag_${flag}
    else
        echo "using char model"
        ttsexpdir=~/disk2/result/librispeech_newdata/exp/tts_baseline_char
        #ttsexpdir=$result_prefix/exp/tts_baseline_char
    fi
    ngpu=1
else
    ttsdata_prefix=/blob/t-fyue/asrtts/libritts_newfeat_maxframes_2000
    result_prefix=/blob/t-fyue/asrtts/asrtts_result
    utt2spk=$ttsdata_prefix/data/train_clean_460/spk_recog_utt2spk
    if [ $spk_loss == "CE" ]; then
        spkexpdir=/blob/t-fyue/asrtts/spkasrtts_results/speaker_model/no_max_len
        spk_model=$spkexpdir/model000000155.model
    else
        spkexpdir=/blob/t-fyue/asrtts/spkasrtts_results/speaker_model/spk_ver
        spk_model=$spkexpdir/model000000029.model
    fi
    unpaireddata_prefix=/blob/t-fyue/asrtts/ted_results 
    #asrexpdir=$result_prefix/exp/asr_baseline_agu_false_newencoder_decode-1_larger_epoch25
    if [ $use_aug == "false" ]; then
        asrexpdir=$result_prefix/exp/asr_baseline_agu_false_newencoder_larger_epoch25
    else
        #asrexpdir=$result_prefix/exp/asr_baseline_agu_true_newencoder_decode-2_larger_epoch25_espnet_new
        if [ ${train_mode} == 1 ];then
            #asrexpdir=$result_prefix/exp/asr_baseline_agu_true_newencoder_decode-2_larger_epoch25_distributed_mask_zero_sparse_image_warp
            if [ ${iterative_training} == 'false' ];then
                asrexpdir=/blob/t-fyue/asrtts/ted_results/exp/spkasrtts_asr-batch_18_tts-batch_45_unpaired-batch_45_unpaired-use-aug_false_use-spk_false_shuffle-spk_false_spk-loss_dist_flag_update-ted-no-spk-1_mode_1
            else
                asrexpdir=/blob/t-fyue/asrtts/ted_results/exp/spkasrtts_asr-batch_18_tts-batch_45_unpaired-batch_45_unpaired-use-aug_false_use-spk_false_shuffle-spk_false_spk-loss_dist_flag_update-ted-only_update_asr_mode_1
            fi
            #asrexpdir=$result_prefix/exp/asr_baseline_agu_true_test_aug_false_newencoder_larger_epoch45_libritts_cmvn
        else
            asrexpdir=$result_prefix/exp/asr_baseline_agu_true_newencoder_decode-2_larger_epoch25_distributed_mask_zero_sparse_image_warp_phone
        fi
    fi
    if [ ${train_mode} == 1 ]; then
        echo "using phone model"
        ttsexpdir=/blob/t-fyue/asrtts/libritts_newfeat_maxframes_2000/exp/train_clean_460_pytorch_phone-lr-decay_16k_maxframe_2000_guided_reduction_factor1_batch128_gpu2-true
        #ttsexpdir=/blob/t-fyue/asrtts/libritts_newfeat_maxframes_2000/exp/train_clean_460_pytorch_phone_lr-decay_16k_maxframe_2000_use-guided_reduction_factor1_batch128_gpu8_ddp_new_spk
        tag=asr-batch_${asr_batch_size}_tts-batch_${tts_batch_size}_unpaired-batch_${unpaired_batch_size}_unpaired-use-aug_${unpaired_aug}_use-spk_${use_spk}_shuffle-spk_${shuffle_spk}_spk-loss_${spk_loss}_flag_${flag}
    else
        echo "using char model"
        ttsexpdir=/blob/t-fyue/asrtts/libritts_newfeat_maxframes_2000/exp/train_clean_460_pytorch_phone_lr-decay_16k_maxframe_2000_guided_reduction_factor1_batch128_gpu2_add_space_espnet_new/
        tag=asr2tts_use-kl_${use_kl}_use-unpaired_${use_unpaired}_batch_${batch_size}_baseline_parallel_${parallel_mode}_aug_${use_aug}_shuffle_spk_${shuffle_spk}_${flag}
    fi   
    ngpu=$gpu_server
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_paired_set=train_clean_460
train_unpaired_set=train
train_dev=dev
recog_set="test_clean"
spk_vector=$result_prefix/exp/xvector_nnet_1a

nj=6
dev_set=$train_dev
eval_set=test_clean
fbankdir=$result_prefix/fbank
dumpdir=$result_prefix/dump   # directory to dump full features
feat_tr_dir=${dumpdir}/${train_set}; 
feat_tr_p_dir=${dumpdir}/${train_paired_set};
feat_tr_up_dir=${unpaireddata_prefix}/dump/${train_unpaired_set}/delta_false;
feat_dt_dir=${unpaireddata_prefix}/dump/$train_dev/delta_false;
feat_ev_dir=${dumpdir}/${eval_set};
asr_dict=$result_prefix/data/lang_char/${train_set}_units.txt

nlsyms=$result_prefix/data/lang_char/non_lang_syms.txt
bpemodel=$result_prefix/data/lang_char/${train_set}_${bpemode}${nbpe}
nnet_dir=$result_prefix/exp/xvector_nnet_1a






asr_model_conf=$asrexpdir/results/model.json
asr_model=$asrexpdir/results/model.acc.best
tts_model=$ttsexpdir/results/model.loss.best
tts_model_conf=$ttsexpdir/results/model.json

#spk_model=

echo $spk_model

echo "stage 5: ASR-TTS training, decode and synthesize"

if [ $(hostname) == stcgpu-17 ]; then
    asrttsexpdir=$result_prefix/exp/spkasrtts_${tag}
else
    asrttsexpdir=/blob/t-fyue/asrtts/ted_results/exp/spkasrtts_${tag}
fi


test_path=$result_prefix/dump/train_small

if [ ${use_aug} == "false" ];then
    preprocess_config=
fi
echo "using prpreprocess_config is ${preprocess_config}"


# if [ ! -s  ${feat_tr_p_dir}/data_tts.json ]; then
#     cp ${feat_tr_p_dir}/data.json ${feat_tr_p_dir}/data_tts.json
#     local/updata_json.sh ${feat_tr_p_dir}/data_tts.json ${nnet_dir}/xvectors_${train_set}/xvector.scp
# fi

# if [ ! -s  ${feat_tr_up_dir}/data_tts.json ]; then
#     cp ${feat_tr_up_dir}/data.json ${feat_tr_up_dir}/data_tts.json
#     local/updata_json.sh ${feat_tr_up_dir}/data_tts.json ${nnet_dir}/xvectors_${train_set}/xvector.scp
# fi

echo $train_mode
if [ $train_mode == 0 ]; then
    # when using paired data, we need to add punctuationed text
    # format : input : target 1 speech
    #                  traget 2 speaker
    #          output: target 1 asr text
    #                  target 2 tts text
    # when we use unpaired data, we plan remove text in training but find it is useful when debugging 
    # format : input : target 1 speach
    #                  target 2 speaker
    #          output: target 1 text
    # notice: the we should correspond id between asr prediction and tts input
    # if [ ! -f ${feat_tr_up_dir}/data_speech.json ];then
    #     python local/transform.py ${feat_tr_up_dir}/data_tts.json ${feat_tr_up_dir}/data_speech.json
    #     # python local/add_char.py ${feat_tr_up_dir}/data_speech.json  ${feat_tr_up_dir}/data.json \
    #     #                       ${feat_tr_up_dir}/data_speech_merge.json
    # fi
    asr_dict=$result_prefix/data/lang_phone_asr/${train_set}_units.txt
    preprocess_config=
    if [ ! -f ${feat_tr_p_dir}/data_speech_merge.json ]; then
        cp ${feat_tr_up_dir}/data_phone_space.json ${feat_tr_up_dir}/data_phone_space_tts.json
        local/updata_json.sh ${feat_tr_up_dir}/data_phone_space_tts.json ${nnet_dir}/xvectors_${train_set}/xvector.scp
        python local/add_phone.py ${feat_tr_p_dir}/data_phone_space_tts.json ${feat_tr_p_dir}/data_space_changeid_tts.json \
                                        ${feat_tr_p_dir}/data_speech_merge.json
    fi
    tr_json_list="${feat_tr_up_dir}/data_phone_space_joint.json ${feat_tr_p_dir}/data_speech_merge.json "
    if [ ${test} == true ];then
        tr_json_list="${test_path}/data_tts.json ${test_path}/data_speech_merge.json"
    fi
elif [ $train_mode == 1 ]; then
    # paired data
    # format : input:  target 1 speech
    #                  target 2 speaker
    #          output: target 1 asr text
    #                  target 2 tts text
    # unpaired data
    # format : input:  target 1 speaker
    #          output: target 2 speech
    
    if [ ! -f ${feat_tr_up_dir}/data_tts_spk.json ]; then
    python local/remove_speech.py ${feat_tr_up_dir}/data.json ${feat_tr_up_dir}/data_text.json
    python local/add_phone.py ${feat_tr_up_dir}/data_text.json  ${feat_tr_up_dir}/data_phone.json  \
                              ${feat_tr_up_dir}/data_tts_spk.json
    fi
    # if [ ! -f ${tts_paired_data} ]; then
    # python local/add_phone.py ${feat_tr_up_dir}/data_tts.json  ${feat_tr_up_dir}/data_phone.json  \
    #                           ${feat_tr_up_dir}/data_tts_spk.json
    # python local/change_speaker.py ${feat_tr_up_dir}/data_tts_spk.json \
    #                                 $ttsdata_prefix/dump/train_clean_460/data_phone_tts_new_spk.json \
    #                                 ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json
    #tts_paired_data=$ttsdata_prefix/dump/train_clean_460/data_phone_tts_new_spk.json
    tts_paired_data=$ttsdata_prefix/dump/train_clean_460/data_phone_tts.json
    asr_paired_data=${feat_tr_p_dir}/data.json
    #python local/change_speaker.py ${feat_tr_up_dir}/data_tts_spk.json $tts_paired_data  ${feat_tr_up_dir}/data_asr2tts_spk.json
    #unpaired_data=${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json
    if [ ${shot_num} == 1 ];then
        # python local/change_speaker_feat.py  ${unpaireddata_prefix}/exp/xvector_nnet_1a/xvectors_test_one_shot/xvector.scp \
        #                                 ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json \
        #                                 ${feat_tr_up_dir}/data_one_shot_adap.json 
        unpaired_data=${feat_tr_up_dir}/data_one_shot_adap.json
    elif [ ${shot_num} == 5 ];then
        # python local/change_speaker_feat.py ${unpaireddata_prefix}/exp/xvector_nnet_1a/xvectors_test_one_shot/xvector.scp \
        #                                 ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json \
        #                                 ${feat_tr_up_dir}/data_five_shot_adap.json 
        unpaired_data=${feat_tr_up_dir}/data_five_shot_adap.json
    elif [ ${shot_num} == "sampling" ];then
        python local/change_speaker_feat.py ${unpaireddata_prefix}/exp/xvector_nnet_1a/xvectors_test_five_sampling/xvector.scp \
                                        ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json \
                                        ${feat_tr_up_dir}/data_five_sampling_adap.json
        unpaired_data=${feat_tr_up_dir}/data_five_sampling_adap.json
    elif [ ${shot_num} == "sampling_400" ];then
        # python local/change_speaker_feat.py ${unpaireddata_prefix}/exp/spk_embedding/test_libritts_cmvn_sampling/spk_vector_400.scp \
        #                                 ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json \
        #                                 ${feat_tr_up_dir}/data_asr2tts_ted_five_sampling_400.json
        unpaired_data=${feat_tr_up_dir}/data_asr2tts_ted_five_sampling_400.json
    elif [ ${shot_num} == "training_spk" ];then
        # python local/change_speaker_feat.py ${unpaireddata_prefix}/exp/spk_embedding/train_libritts_cmvn/spk_vector.scp \
        #                                 ${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json \
        #                                 ${feat_tr_up_dir}/data_tts_training_spk.json
        unpaired_data=${feat_tr_up_dir}/data_tts_training_spk.json
    else
        unpaired_data=${feat_tr_up_dir}/data_asr2tts_libritts_new_spk.json
    fi
    #unpaired_data=${feat_tr_up_dir}/data_tts_merge.json
    #unpaired_data=${test_path}/unpaired_text_small_1.json
    echo $train_mode
    tr_json_list="${unpaired_data} ${asr_paired_data} ${tts_paired_data} "
    if [ ${test} == true ];then
        tr_json_list="${test_path}/unpaired_text_small.json ${test_path}/asr_paired_small.json ${test_path}/tts_paired_small.json"
    fi
fi




asr_dev_json=${feat_dt_dir}/data.json
#tts_dev_json=$ttsdata_prefix/dump/dev_clean/data_phone_tts_new_spk.json
tts_dev_json=$ttsdata_prefix/dump/dev_clean/data_phone_tts.json
if [ ${test} == true ];then
    asr_dev_json=${test_path}/asr_dev_small.json
    tts_dev_json=${test_path}/tts_dev_small.json
fi
asrttsexpdir=${asrttsexpdir}_mode_${train_mode}


resume=
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

# echo "copy asr conf to ${asrttsexpdir}/results"
# cp $asr_model_conf ${asrttsexpdir}/results


# 0 train_other_speech + proposed grapheme TTS + pretrain 460 ASR
# 1 train_other_text + proposed phoneme TTS + pretrain 460 ASR
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
    #${cuda_cmd} --gpu 1 ${asrttsexpdir}/train.log \
    $train_cmd \
    --config ${train_asrtts_config} \
    --preprocess-conf ${preprocess_config} \
    --ngpu $ngpu \
    --backend ${backend} \
    --outdir ${asrttsexpdir}/results \
    --tensorboard-dir $result_prefix/tensorboard/asrtts_${tag} \
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
    --spk-model ${spk_model} \
    --utt2spk ${utt2spk} \
    --asr2tts-policy ${asr2tts_policy} \
    --parallel-mode ${parallel_mode} \
    --shuffle-spk ${shuffle_spk} \
    --asr-batch-size ${asr_batch_size} \
    --tts-batch-size ${tts_batch_size} \
    --unpaired-batch-size ${unpaired_batch_size} \
    --update-asr ${update_asr} \
    --update-tts ${update_tts} \
    --update-tts2asr ${update_tts2asr} \
    --update-spk ${update_spk} \
    --use-spk ${use_spk} \
    --use-inference ${use_inference} \
    --unpaired-aug ${unpaired_aug} \
    --spk-loss ${spk_loss} \
    --mix-precision ${mix_precision} \
    --train-mode $train_mode 
fi