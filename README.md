# Exploring Machine Speech Chain for Domain Adaptation and Few-Shot Speaker Adaptation
This is an implementation of the [paper](https://arxiv.org/abs/2104.03815), based on the [ESPnet](https://github.com/espnet/espnet). 
If you have any questions, please email to me(11930381@mail.sustech.edu.cn).
# Requirements
Follow the [installation](https://espnet.github.io/espnet/installation.html) method of espnet.  
You should use torch==1.7.1.
# Pretraining
You should download [LibriSpeech](http://www.openslr.org/12/) and [LibriTTS](http://www.openslr.org/60/) manually.
LibriSpeech: run egs/librispeech/asr/pretrain_asr.sh(The recipe train ASR model on LibriSpeech train-clean-460)  
LibriTTS: run egs/libritts/tts/pretrain_tts.sh(The recipe train ASR model on LibriTTS train-clean-460)
# Joint training
You should download [TED-LIUM-1](http://www.openslr.org/7/) manually.
We give the punctuated TED_LIUM  text under egs/tedlium/data path.  
Go to egs/tedlium.
Run ./prepare_data.sh for preparing json file for training, and then run ./joint_training.sh.
