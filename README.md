# Machine Speech Chain for Domain Adaptation and Few-Shot Speaker Adaptation
This is an implementation of the paper, based on the [ESPnet](https://github.com/espnet/espnet). 
If you have any questions, please email to me(11930381@mail.sustech.edu.cn).
# Requirements
Follow the [installation](https://espnet.github.io/espnet/installation.html) method of espnet.  
torch==1.7.1
# Pretraining
You should download [LibriSpeech](http://www.openslr.org/12/) and [LibriTTS](http://www.openslr.org/60/) manually.  
Go to egs/librispeech/asr1 and egs/libritts/tts, run ./pretrain_asr.sh and ./pretrain_tts.sh
# Joint training
You should download [TED-LIUM-1](http://www.openslr.org/7/) manually.
We give the punctuated TED_LIUM  text under egs/tedlium/data path.  
Go to egs/tedlium.
Run ./prepare_data.sh for preparing json file for training, and then run ./joint_training.sh.

