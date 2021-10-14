# Exploring Machine Speech Chain for Domain Adaptation
This is an implementation of the paper, based on the [ESPnet](https://github.com/espnet/espnet). 
If you have any questions, please email to me(11930381@mail.sustech.edu.cn).
# Requirements
Follow the [installation](https://espnet.github.io/espnet/installation.html) method of espnet.  
You should use torch==1.7.1.
# Pretraining
You should download [LibriSpeech](http://www.openslr.org/12/) and [LibriTTS](http://www.openslr.org/60/) manually.  
LibriSpeech: run ./pretrain_asr.sh under egs/librispeech/asr (The recipe train ASR model on LibriSpeech train-clean-460)  
LibriTTS: run ./pretrain_tts.sh under egs/libritts/tts (The recipe train TTS model on LibriTTS train-clean-460)
# Joint training
You should download [TED-LIUM-1](http://www.openslr.org/7/) manually.
We give the punctuated TED_LIUM  text under egs/tedlium/data path.  
Execution directory(egs/tedlium/asrtts):  
Run ./prepare_data.sh for preparing json file for training, and then run ./joint_training.sh for joint training.
## Experimental options in joint_training.sh for three-stage training
### Stage 1:
update_asr=true  
update_tts=false  
update_tts2asr=true  
filter_data=true  
filter_thre=0.58   
unpaired_aug=true   

### Stage 2:
asrexpdir=    # change the path of asr baseline to the asr adaptation
update_asr=false
update_tts=true
update_tts2asr=true
filter_data=false
unpaired_aug=flase
tts_loss_weight=0.005

### Stage 3:
ttsexpdir=  # change the path of tts baseline to the asr adaptation
update_asr=false
update_tts=true
update_tts2asr=true
filter_data=true
filter_thre=0.58 
unpaired_aug=true 


