#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import random
import sys

import editdistance

import chainer
import numpy as np
import pickle
from numpy.lib.function_base import select
import six
import torch
from torch import tensor
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
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list

# from espnet.utils.spec_augment import specaug
from espnet2.asr.specaug.specaug import SpecAug
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
        self.spec = SpecAug(apply_time_warp=False,freq_mask_width_range=(0,30),time_mask_width_range=(0,40))
        self.weight = weight
        self.shuffle_spk = args.shuffle_spk
        self.args = args
        self.generator = args.generator
    #xs_tts, ilens_tts, ys_asr,spembs, zero_att=self.zero_att
    def forward(self, asr_batch,tts_batch,unpaired_batch, iteration=0, use_spk=True, model_device=None, unpaired_augstep=0,asr_mode=1):
        """TACOTRON2 LOSS FORWARD CALCULATION

        :param torch.Tensor xs_tts: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: loss value
        :rtype: torch.Tensor
        """
        tts_asr_loss = 0
        tts_asr_loss_data = 0
        acc = 0
        if self.args.update_tts2asr:
            # unpaired data
            unpaired_ys_pad, ys_tts_pad, asr_char_olens,tts_char_olens, unpaired_spembs = self.unpaired_converter(unpaired_batch,model_device)
            # generate feature sequences for a batch
            if self.args.shuffle_spk:
                unpaired_spembs,spk_index = shuffle_spk(unpaired_spembs)
            logging.info("the length text of input tts model during tts->asr loss:"+ str(tts_char_olens.tolist()))
            if self.args.update_tts:
                self.tts_model.train()
                after_outs, logits, flens, filter_index = self.tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs)
                now_unpaired_ys_pad = unpaired_ys_pad
            else:
                self.tts_model.eval()
                with torch.no_grad():
                    after_outs, logits, flens, filter_index = self.tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs,self.args.filter_data,self.args.filter_thre)

                    if self.args.unpaired_aug:
                        after_outs, _= self.spec(after_outs,flens)
                        #logging.info(after_outs.size())
                    now_asr_char_olens = asr_char_olens[filter_index]
                    now_unpaired_ys_pad = unpaired_ys_pad[filter_index]

            logging.info("feak data length: " + str(flens))
            if self.generator == 'tts':
                #flens = torch.tensor(flens,dtype=now_tts_char_olens.dtype)
                flens,indice = torch.sort(flens,descending=True)
                after_outs = after_outs[indice]
                now_unpaired_ys_pad = now_unpaired_ys_pad[indice]
                if flens.size(0)>0:
                    enc_outs, enc_flens, _ = self.asr_model.enc(after_outs, flens)

            logging.info("feak data prediction")
      
            if flens.size(0)>0:
                tts_asr_loss,acc,ppl = self.asr_model.dec(enc_outs, enc_flens, now_unpaired_ys_pad)
            else:
                tts_asr_loss = torch.FloatTensor([0]).to(now_unpaired_ys_pad.device)
                acc = 1.0
            tts_asr_loss_data = tts_asr_loss.detach().cpu().numpy()
            logging.info("tts->asr loss = %.3e " % tts_asr_loss_data)
        if self.reporter is not None:
            self.reporter.report(None, None, tts_asr_loss_data, None, acc, None, None)
        if self.args.update_asr:
            asr_xs_pad, asr_ilens, asr_ys_pad = self.asr_converter(asr_batch,model_device)
            asr_loss = self.asr_model(asr_xs_pad, asr_ilens, asr_ys_pad)

        else:
            asr_loss = None

        if asr_loss is None:
            loss = tts_asr_loss
        else:   
            loss = tts_asr_loss + asr_loss
        if tts_batch:
            xs_pad, ilens, ys_pad, labels, olens, spembs= self.tts_converter(tts_batch,model_device)
            tts_loss = self.tts_model(xs_pad, ilens, ys_pad, labels, olens, spembs)
            loss = self.args.tts_loss_weight * tts_asr_loss + tts_loss
        
        self.reporter.report(None, float(ppl), tts_asr_loss_data, None, acc, None, None)
        logging.info("total loss is %f" % loss.detach().cpu().numpy())
        return loss


class Tacotron2ASRLoss_2(torch.nn.Module):
    """TACOTRON2 ASR-LOSS FUNCTION

    :param torch.nn.Module model: tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    :param bool report: Use reporter to log loss values (deafult true)
    :param bool reduce: Reduce the loss over the batch size
    """

    def __init__(self,tts_model,last_tts_model, asr_model, lm_model, args, asr_converter, tts_converter, unpaired_converter, reporter=None, weight=1.0):
        super(Tacotron2ASRLoss_2, self).__init__()
        self.tts_model = tts_model
        self.asr_model = asr_model
        self.lm_model = lm_model
        if lm_model:
            self.kl_criterion = torch.nn.KLDivLoss(reduction='sum')
        self.last_tts_model = last_tts_model
        self.asr_converter = asr_converter
        self.tts_converter = tts_converter
        self.unpaired_converter = unpaired_converter
        self.reporter = reporter
        self.spec = SpecAug(apply_time_warp=False,freq_mask_width_range=(0,30),time_mask_width_range=(0,40))
        self.weight = weight
        self.shuffle_spk = args.shuffle_spk
        self.args = args
        self.generator = args.generator
    #xs_tts, ilens_tts, ys_asr,spembs, zero_att=self.zero_att
    def forward(self, asr_batch,tts_batch,unpaired_batch, iteration=0, use_spk=True, model_device=None, unpaired_augstep=0,asr_mode=1):
        """TACOTRON2 LOSS FORWARD CALCULATION

        :param torch.Tensor xs_tts: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: loss value
        :rtype: torch.Tensor
        """
        tts_asr_loss = 0
        tts_asr_loss_data = 0
        acc = 0
        
        # unpaired data
        unpaired_ys_pad, ys_tts_pad, asr_char_olens,tts_char_olens, unpaired_spembs = self.unpaired_converter(unpaired_batch,model_device)
        # generate feature sequences for a batch
        if self.args.shuffle_spk:
            unpaired_spembs,spk_index = shuffle_spk(unpaired_spembs)
        logging.info("the length text of input tts model during tts->asr loss:"+ str(tts_char_olens.tolist()))
        
        self.tts_model.train()
        after_outs, logits, flens, filter_index = self.tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs)
                 
        if self.last_tts_model:
            self.last_tts_model.eval()
            with torch.no_grad():
                last_after_outs, last_logits, last_flens, last_filter_index = self.last_tts_model.model.generate(ys_tts_pad, tts_char_olens, unpaired_spembs,self.args.filter_data,self.args.filter_thre)

            # for asr loss
            asr_flens = flens[~last_filter_index]
            asr_char_olens = asr_char_olens[~last_filter_index]
            asr_after_outs = after_outs[~last_filter_index]
            asr_unpaired_ys_pad = unpaired_ys_pad[~last_filter_index]

            logging.info(last_after_outs.size(0))
            # for tts loss
            tts_xs_pad = ys_tts_pad[last_filter_index]
            tts_ilens = tts_char_olens[last_filter_index]
            tts_ys_pad = last_after_outs[last_filter_index]
            tts_labels = tts_ys_pad.new_zeros(tts_ys_pad.size(0), tts_ys_pad.size(1))
            tts_olens= last_flens[last_filter_index]
            tts_spembs = unpaired_spembs[last_filter_index]
            for i, l in enumerate(tts_olens):
                tts_labels[i, l - 1 :] = 1.0
            spembs = unpaired_spembs[last_filter_index]

        logging.info("feak data length: " + str(flens))
        if asr_flens.size(0) > 0:
            if self.generator == 'tts':
                #flens = torch.tensor(flens,dtype=now_tts_char_olens.dtype)
                asr_flens,indice = torch.sort(asr_flens,descending=True)
                asr_after_outs = asr_after_outs[indice]
                asr_unpaired_ys_pad = asr_unpaired_ys_pad[indice]

            logging.info("feak data prediction")
            # for asr loss
            enc_outs, enc_flens, _ = self.asr_model.enc(asr_after_outs, asr_flens)
            tts_asr_loss,acc,ppl = self.asr_model.dec(enc_outs, enc_flens, asr_unpaired_ys_pad)
            tts_asr_loss_data = tts_asr_loss.detach().cpu().numpy()
            logging.info("tts->asr loss = %.3e " % tts_asr_loss_data)
        else:
            tts_asr_loss = 0
            tts_asr_loss_data = 0
            acc = 1
            ppl = 1

        # for fake tts loss
        if tts_ilens.size(0) > 0:
            fake_tts_loss = self.tts_model(tts_xs_pad, tts_ilens, tts_ys_pad, tts_labels, tts_olens, tts_spembs,fake_loss=True)
        else:
            fake_tts_loss = 0
        logging.info("fake tts loss:%f" % float(fake_tts_loss))
        if tts_batch:
            xs_pad, ilens, ys_pad, labels, olens, spembs= self.tts_converter(tts_batch,model_device)
            tts_loss = self.tts_model(xs_pad, ilens, ys_pad, labels, olens, spembs)
            if float(tts_asr_loss) > 30:
                #tts_asr_loss = 0
                logging.warning("the asr2tts loss is %f, there maybe some errors" % float(tts_asr_loss))
            loss = self.args.tts_loss_weight * tts_asr_loss + tts_loss + fake_tts_loss

        self.reporter.report(None, float(ppl), tts_asr_loss_data, None, acc, None, None)
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
    if "taco2_loss.bce_criterion.pos_weight" in model_state_dict.keys():
        model_state_dict.pop("taco2_loss.bce_criterion.pos_weight")
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict

def TacotronRewardLoss(tts_model_file, idim=None, odim=None, train_args=None,
                       use_masking=True, bce_pos_weight=1.0, reporter=None,train_mode=None,asr2tts_policy=None):
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
    # Load loss
    return TacotronRewardLoss(
        tts_model_file,
        idim=idim_taco,
        odim=odim_taco,
        train_args=train_args_taco,
        reporter=reporter,
        train_mode = train_mode,
        asr2tts_policy=asr2tts_policy
    ),train_args_taco
