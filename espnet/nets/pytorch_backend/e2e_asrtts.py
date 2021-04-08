#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import random

import editdistance

import chainer
import numpy as np
import pickle
import six
import torch
import torch.nn.functional as F

# from chainer import reporter

from espnet.asr.asr_utils import torch_load
from espnet.nets.e2e_asr_common import label_smoothing_dist

from espnet.nets.pytorch_backend.rnn.attentions import att_for
# from espnet.nets.pytorch_backend.rnn.decoders_asrtts import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

# from espnet.nets.pytorch_backend.nets_utils import mask_by_length_and_multiply
from espnet.nets.pytorch_backend.nets_utils import pad_list
# from espnet.nets.pytorch_backend.nets_utils import set_requires_grad
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

from espnet.nets.pytorch_backend.e2e_tts import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts import Tacotron2Loss

# from espnet.utils.spec_augment import specaug
# from espnet2.asr.specaug.specaug import SpecAug
# from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2
# from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss

CTC_LOSS_THRESHOLD = 10000
grads={}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
        logging.info(grad)
    return hook


def shuffle_spk(spembs):
    """
    :param torch.Tensor spembs: batch of speaker embedding sequnece tenser (B)
    """

    index = [ i for i in range(spembs.size(0)) ]
    random.shuffle(index)
    logging.info("after shuffle, the spk embed index is:"+str(index))
    spembs = spembs[index]
    return spembs,index


class Tacotron2ASRLoss(torch.nn.Module):
    """TACOTRON2 ASR-LOSS FUNCTION

    :param torch.nn.Module model: tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    :param bool report: Use reporter to log loss values (deafult true)
    :param bool reduce: Reduce the loss over the batch size
    """

    def __init__(self,tts_model, asr_model, args, asr_converter, tts_converter, unpaired_converter, reporter=None, weight=1.0):
        super(Tacotron2ASRLoss, self).__init__()
        self.tts_model = tts_model
        self.asr_model = asr_model
        self.asr_converter = asr_converter
        self.tts_converter = tts_converter
        self.unpaired_converter = unpaired_converter
        self.reporter = reporter
        self.weight = weight
        self.shuffle_spk = args.shuffle_spk
        self.args = args
        self.generator = args.generator
    #xs_tts, ilens_tts, ys_asr,spembs, zero_att=self.zero_att
    def forward(self, asr_batch,tts_batch,unpaired_batch, iteration=0, use_spk=True, model_device=None, unpaired_augstep=0,after_outs=None):
        """TACOTRON2 LOSS FORWARD CALCULATION

        :param torch.Tensor xs_tts: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: loss value
        :rtype: torch.Tensor
        """
        # unpaired data
        unpaired_ys_pad, ys_tts_pad, asr_char_olens,tts_char_olens, unpaired_spembs = self.unpaired_converter(unpaired_batch,model_device)
        # generate feature sequences for a batch
        if self.args.shuffle_spk:
            unpaired_spembs,spk_index = shuffle_spk(unpaired_spembs)
        logging.info("the length text of input tts model during tts->asr loss:"+ str(tts_char_olens.tolist()))
        if self.args.update_tts:
            self.tts_model.train()
            after_outs, logits, flens, filter_indice = self.tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs)
        else:
            self.tts_model.eval()
            with torch.no_grad():
                after_outs, logits, flens, filter_indice = self.tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs)
        logging.info("feak data length: " + str(flens))
        if self.generator == 'tts':
            flens = torch.tensor(flens,dtype=tts_char_olens.dtype)
            flens,indice = torch.sort(flens,descending=True)
            after_outs = after_outs[indice]
            unpaired_ys_pad = unpaired_ys_pad[indice]
            enc_outs, enc_flens, _ = self.asr_model.enc(after_outs, flens)
        logging.info("feak data prediction")
        tts_asr_loss,acc,ppl = self.asr_model.dec(enc_outs, enc_flens, unpaired_ys_pad)
        tts_asr_loss_data = tts_asr_loss.detach().cpu().numpy()
        logging.info("tts->asr loss = %.3e " % tts_asr_loss_data)
        if self.reporter is not None:
            self.reporter.report(None, tts_asr_loss_data, None, None, None, None)
        if self.args.update_asr:
            asr_xs_pad, asr_ilens, asr_ys_pad = self.asr_converter(asr_batch,model_device)
            asr_loss = self.asr_model(asr_xs_pad, asr_ilens, asr_ys_pad)
        else:
            asr_loss = None


        if asr_loss is None:
            loss = tts_asr_loss
        else:
            loss = tts_asr_loss + asr_loss

        logging.info("total loss is %f" % loss.detach().cpu().numpy())
        return loss


def load_tts(path,model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']

    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_state_dict.pop("taco2_loss.bce_criterion.pos_weight")
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict

def TacotronRewardLoss(tts_model_file, idim=None, odim=None, train_args=None,
                       use_masking=False, bce_pos_weight=1.0,
                       spk_embed_dim=None, update_asr_only=True, reporter=None,train_mode=None,asr2tts_policy=None):
    # TACOTRON CYCLE-CONSISTENT LOSS HERE
    # Define model
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        args=train_args,
        train_mode=train_mode,
        asr2tts_policy=asr2tts_policy
    )
    if tts_model_file:
        # load trained model parameters
        logging.info('reading model parameters from ' + tts_model_file)
        load_tts(tts_model_file,tacotron2)
        #torch_load(tts_model_file, tacotron2)
    else:
        logging.info("not using pretrained tacotron2 model")
    # Define loss
    loss = Tacotron2Loss(
        model=tacotron2,
        use_masking=use_masking,
        bce_pos_weight=bce_pos_weight,
        reporter=reporter,
        train_mode=train_mode,
        asr2tts_policy=asr2tts_policy
    )
    loss.train_args = train_args
    return loss


def load_tacotron_loss(tts_model_conf, tts_model_file, args, reporter=None,train_mode=None,asr2tts_policy=None):
    # Read model
    if 'conf' in tts_model_conf:
        with open(tts_model_conf, 'rb') as f:
            idim_taco, odim_taco, train_args_taco = pickle.load(f)
    elif 'json' in tts_model_conf:
        from espnet.asr.asr_utils import get_model_conf
        idim_taco, odim_taco, train_args_taco = get_model_conf(tts_model_file, conf_path=tts_model_conf)
    if args.modify_output:
        import json
        with open(args.valid_json, 'rb') as f:
            valid_json = json.load(f)['utts']
        utts = list(valid_json.keys())
        idim_taco = int(valid_json[utts[0]]['output'][0]['shape'][1])
        from espnet.asr.asrtts_utils import remove_output_layer
        pretrained_model = remove_output_layer(torch.load(tts_model_file),
                                               idim_taco, args.eprojs, train_args_taco.embed_dim, 'tts')

        torch.save(pretrained_model, 'tmp.model')
        tts_model_file = 'tmp.model'
    # Load loss
    return TacotronRewardLoss(
        tts_model_file,
        idim=idim_taco,
        odim=odim_taco,
        train_args=train_args_taco,
        update_asr_only=args.update_asr_only,
        reporter=reporter,
        train_mode = train_mode,
        asr2tts_policy=asr2tts_policy
    ),train_args_taco
