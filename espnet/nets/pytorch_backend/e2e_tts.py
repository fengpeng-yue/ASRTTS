#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import six

import chainer
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc


from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask as make_mask
from espnet.nets.pytorch_backend.asrtts_tacotron.encoder import Encoder
from espnet.nets.pytorch_backend.asrtts_tacotron.decoder import Decoder
from espnet.nets.pytorch_backend.asrtts_tacotron.cbhg import CBHG
from espnet.nets.tts_interface import TTSInterface
#from torch.cuda.amp import autocast as autocast

grads={}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
        #logging.info(grad)
        logging.info(grad.abs().mean())
    return hook


class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.

    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.

    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969

    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.

        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.

        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.

        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.

        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])

        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


def encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('tanh'))


def make_non_pad_mask(lengths):
    """Function to make tensor mask containing indices of the non-padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[1, 1, 1, 1 ,1],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of non-padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand < seq_length_expand


class Reporter(chainer.Chain):
    def report(self, dicts):
        for d in dicts:
            chainer.reporter.report(d, self)





class Tacotron2Loss(torch.nn.Module):
    """Tacotron2 loss function

    :param torch.nn.Module model: tacotron2 model
    :param bool use_masking: whether to mask padded part in loss calculation
    :param float bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    """

    def __init__(self, model, use_guided_attn_loss=True,use_masking=True, bce_pos_weight=1.0, reporter=None,
                train_mode=None,asr2tts_policy=None):
        super(Tacotron2Loss, self).__init__()
        self.model = model
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight
        self.train_mode = train_mode
        self.asr2tts_policy = asr2tts_policy
        if hasattr(model, 'module'):
            self.use_cbhg = model.module.use_cbhg
            self.reduction_factor = model.module.reduction_factor
        else:
            self.use_cbhg = model.use_cbhg
            self.reduction_factor = model.reduction_factor
        if reporter is None:
            self.reporter = Reporter()
        else:
            self.reporter = reporter
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
            sigma=self.model.args.guided_attn_loss_sigma,
            alpha=self.model.args.guided_attn_loss_lambda,
        )
        

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, spcs=None,
                softargmax=False):
        """Tacotron2 loss forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor labels: batch of the sequences of stop token labels (B, Lmax)
        :param list olens: batch of the lengths of each target (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :param torch.Tensor spcs: batch of padded target features (B, Lmax, spc_dim)
        :return: loss value
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # calcuate outputs
        if self.use_cbhg:
            cbhg_outs, after_outs, before_outs, logits, att_ws = self.model(xs, ilens, ys, olens, spembs,asr2tts=True)
        else:
            after_outs, before_outs, logits, att_ws = self.model(xs, ilens, ys, olens, spembs, softargmax,asr2tts=True)

        # remove mod part
        if self.reduction_factor > 1:
            olens = [olen - olen % self.reduction_factor for olen in olens]
            ys = ys[:, :max(olens)]
            labels = labels[:, :max(olens)]
            spcs = spcs[:, :max(olens)] if spcs is not None else None

        # prepare weight of positive samples in cross entorpy
        if self.bce_pos_weight != 1.0:
            weights = ys.new(*labels.size()).fill_(1)
            weights.masked_fill_(labels.eq(1), self.bce_pos_weight)
        else:
            weights = None

        # perform masking for padded values
        if self.use_masking:
            mask = to_device(self, make_non_pad_mask(olens).unsqueeze(-1))
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            if self.use_cbhg:
                spcs = spcs.masked_select(mask)
                cbhg_outs = cbhg_outs.masked_select(mask)

        # calculate loss
        # if self.asr2tts_policy == "policy_gradient":
        #     l1_loss = F.l1_loss(after_outs, ys, reduce=False) + F.l1_loss(before_outs,
        #                                                                   ys, reduce=False)
        #     l1_loss = l1_loss.mean(2).mean(1)
        #     mse_loss = F.mse_loss(after_outs, ys, reduce=False) + F.mse_loss(before_outs,
        #                                                                      ys, reduce=False)
        #     mse_loss = mse_loss.mean(2).mean(1)
        #     bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights, reduce=False)
        #     bce_loss = bce_loss.mean(1)
        # else:
        l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)
            
        loss = l1_loss + mse_loss + bce_loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi):
            # length of output for auto-regressive input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            self.reporter.report3(loss.detach().cpu().numpy(),attn_loss.detach().cpu().numpy(),l1_loss.detach().cpu().numpy(),mse_loss.detach().cpu().numpy(),bce_loss.detach().cpu().numpy())
        else:
            attn_loss = NotImplementedError
            self.reporter.report3(loss.detach().cpu().numpy(),None,l1_loss.detach().cpu().numpy(),mse_loss.detach().cpu().numpy(),bce_loss.detach().cpu().numpy())
        #tts_loss = loss.detach().cpu().numpy()
        
        logging.info("tts loss is: %f" % loss.detach().cpu().numpy())
        
        # if self.use_cbhg:
        #     # calculate chbg loss and then itegrate them
        #     cbhg_l1_loss = F.l1_loss(cbhg_outs, spcs)
        #     cbhg_mse_loss = F.mse_loss(cbhg_outs, spcs)
        #     loss = l1_loss + mse_loss + bce_loss + cbhg_l1_loss + cbhg_mse_loss
        #     # report loss values for logging
        #     self.reporter.report([
        #         {'l1_loss': l1_loss.item()},
        #         {'mse_loss': mse_loss.item()},
        #         {'bce_loss': bce_loss.item()},
        #         {'cbhg_l1_loss': cbhg_l1_loss.item()},
        #         {'cbhg_mse_loss': cbhg_mse_loss.item()},
        #         {'loss': loss.item()}])
        # elif:
            # integrate loss
        #   loss = l1_loss + mse_loss + bce_loss
            # report loss values for logging
            #self.reporter.report(None, None, None, None, None, None, None, mse_loss, None)
            #logging.info(loss.mean(0).item())
        # else:
        #     # integrate loss
        #     loss = l1_loss + mse_loss + bce_loss
        #     # report loss values for logging
        #     self.reporter.report([
        #         {'l1_loss': l1_loss.item()},
        #         {'mse_loss': mse_loss.item()},
        #         {'bce_loss': bce_loss.item()},
        #         {'loss': loss.item()}])
        #     logging.info(loss.item())

        return loss


class Tacotron2(torch.nn.Module):
    """Tacotron2 based Seq2Seq converts chars to features

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param Namespace args: argments containing following attributes
        (int) spk_embed_dim: dimension of the speaker embedding
        (int) embed_dim: dimension of character embedding
        (int) elayers: the number of encoder blstm layers
        (int) eunits: the number of encoder blstm units
        (int) econv_layers: the number of encoder conv layers
        (int) econv_filts: the number of encoder conv filter size
        (int) econv_chans: the number of encoder conv filter channels
        (int) dlayers: the number of decoder lstm layers
        (int) dunits: the number of decoder lstm units
        (int) prenet_layers: the number of prenet layers
        (int) prenet_units: the number of prenet units
        (int) postnet_layers: the number of postnet layers
        (int) postnet_filts: the number of postnet filter size
        (int) postnet_chans: the number of postnet filter channels
        (str) output_activation: the name of activation function for outputs
        (int) adim: the number of dimension of mlp in attention
        (int) aconv_chans: the number of attention conv filter channels
        (int) aconv_filts: the number of attention conv filter size
        (bool) cumulate_att_w: whether to cumulate previous attention weight
        (bool) use_batch_norm: whether to use batch normalization
        (bool) use_concate: whether to concatenate encoder embedding with decoder lstm outputs
        (float) dropout: dropout rate
        (float) zoneout: zoneout rate
        (int) reduction_factor: reduction factor
        (bool) use_cbhg: whether to use CBHG module
        (int) cbhg_conv_bank_layers: the number of convoluional banks in CBHG
        (int) cbhg_conv_bank_chans: the number of channels of convolutional bank in CBHG
        (int) cbhg_proj_filts: the number of filter size of projection layeri in CBHG
        (int) cbhg_proj_chans: the number of channels of projection layer in CBHG
        (int) cbhg_highway_layers: the number of layers of highway network in CBHG
        (int) cbhg_highway_units: the number of units of highway network in CBHG
        (int) cbhg_gru_units: the number of units of GRU in CBHG
    """

    def __init__(self, idim, odim, args,train_mode=None,asr2tts_policy=None):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.args = args
        self.spk_embed_dim = args.spk_embed_dim
        self.embed_dim = args.embed_dim
        self.elayers = args.elayers
        self.eunits = args.eunits
        self.econv_layers = args.econv_layers
        self.econv_filts = args.econv_filts
        self.econv_chans = args.econv_chans
        self.dlayers = args.dlayers
        self.dunits = args.dunits
        self.prenet_layers = args.prenet_layers
        self.prenet_units = args.prenet_units
        self.postnet_layers = args.postnet_layers
        self.postnet_chans = args.postnet_chans
        self.postnet_filts = args.postnet_filts
        self.adim = args.adim
        self.aconv_filts = args.aconv_filts
        self.aconv_chans = args.aconv_chans
        self.cumulate_att_w = args.cumulate_att_w
        self.use_batch_norm = args.use_batch_norm
        self.use_concate = args.use_concate
        self.dropout = args.dropout_rate
        self.zoneout = args.zoneout_rate
        self.reduction_factor = args.reduction_factor
        self.atype = args.atype
        self.use_cbhg = args.use_cbhg
        if self.use_cbhg:
            self.spc_dim = args.spc_dim
            self.cbhg_conv_bank_layers = args.cbhg_conv_bank_layers
            self.cbhg_conv_bank_chans = args.cbhg_conv_bank_chans
            self.cbhg_conv_proj_filts = args.cbhg_conv_proj_filts
            self.cbhg_conv_proj_chans = args.cbhg_conv_proj_chans
            self.cbhg_highway_layers = args.cbhg_highway_layers
            self.cbhg_highway_units = args.cbhg_highway_units
            self.cbhg_gru_units = args.cbhg_gru_units

        # define activation function for the final output
        if args.output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, args.output_activation):
            self.output_activation_fn = getattr(F, args.output_activation)
        else:
            raise ValueError('there is no such an activation function. (%s)' % args.output_activation)
        # define network modules
        self.enc = Encoder(idim=self.idim,
                           embed_dim=self.embed_dim,
                           elayers=self.elayers,
                           eunits=self.eunits,
                           econv_layers=self.econv_layers,
                           econv_chans=self.econv_chans,
                           econv_filts=self.econv_filts,
                           use_batch_norm=self.use_batch_norm,
                           dropout=self.dropout,
                           train_mode=train_mode,
                           asr2tts_policy=asr2tts_policy)
        dec_idim = self.eunits if self.spk_embed_dim is None else self.eunits + self.spk_embed_dim
        if self.atype == "location":
            att = AttLoc(dec_idim,
                         self.dunits,
                         self.adim,
                         self.aconv_chans,
                         self.aconv_filts)
        elif self.atype == "forward":
            att = AttForward(dec_idim,
                             self.dunits,
                             self.adim,
                             self.aconv_chans,
                             self.aconv_filts)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        elif self.atype == "forward_ta":
            att = AttForwardTA(dec_idim,
                               self.dunits,
                               self.adim,
                               self.aconv_chans,
                               self.aconv_filts,
                               self.odim)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(idim=dec_idim,
                           odim=self.odim,
                           att=att,
                           dlayers=self.dlayers,
                           dunits=self.dunits,
                           prenet_layers=self.prenet_layers,
                           prenet_units=self.prenet_units,
                           postnet_layers=self.postnet_layers,
                           postnet_chans=self.postnet_chans,
                           postnet_filts=self.postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=self.use_batch_norm,
                           use_concate=self.use_concate,
                           dropout=self.dropout,
                           zoneout=self.zoneout,
                           reduction_factor=self.reduction_factor
                           )

        if self.use_cbhg:
            self.cbhg = CBHG(idim=self.odim,
                             odim=self.spc_dim,
                             conv_bank_layers=self.cbhg_conv_bank_layers,
                             conv_bank_chans=self.cbhg_conv_bank_chans,
                             conv_proj_filts=self.cbhg_conv_proj_filts,
                             conv_proj_chans=self.cbhg_conv_proj_chans,
                             highway_layers=self.cbhg_highway_layers,
                             highway_units=self.cbhg_highway_units,
                             gru_units=self.cbhg_gru_units)

        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)
    #@autocast()
    def forward(self, xs, ilens, ys, olens=None, spembs=None, softargmax=False,asr2tts=False):
        """Tacotron2 forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens:
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attention weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        hs, hlens = self.enc(xs, ilens, softargmax=softargmax,asr2tts=asr2tts)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)

        if self.use_cbhg:
            if self.reduction_factor > 1:
                olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            return cbhg_outs, after_outs, before_outs, logits
        else:
            return after_outs, before_outs, logits, att_ws

    def inference(self, x, inference_args, spemb=None):
        """Generates the sequence of features given the sequences of characters

        :param torch.Tensor x: the sequence of characters (T)
        :param Namespace inference_args: argments containing following attributes
            (float) threshold: threshold in inference
            (float) minlenratio: minimum length ratio in inference
            (float) maxlenratio: maximum length ratio in inference
        :param torch.Tensor spemb: speaker embedding vector (spk_embed_dim)
        :return: the sequence of features (L, odim)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        # get options
        # threshold = inference_args.threshold
        # minlenratio = inference_args.minlenratio
        # maxlenratio = inference_args.maxlenratio

        threshold = 0.5
        minlenratio = 0
        maxlenratio = 14.0

        # inference
        h = self.enc.inference(x)
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws = self.dec.inference(h, threshold, minlenratio, maxlenratio)

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws
        else:
            return outs, probs, att_ws
    #@autocast()
    def generate(self, xs, ilens, spembs=None, softargmax=False):
        """TACOTRON2 FEATURE GENERATION

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attetion weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))
        hs, hlens = self.enc(xs, ilens, softargmax=softargmax)
        # logging.info(hs)
        # logging.info(hlens)
        if self.spk_embed_dim is not None:
            spembs= spembs.unsqueeze(1).expand(-1, hs.size(1), -1)
            #spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        hlens = hlens.to(hs.device).float()
        outs, logits, ylens,filter_indice = self.dec.generate(hs, hlens)
        mask =  make_mask(ylens).to(outs.device).unsqueeze(-2)
        mask = mask.eq(0).transpose(-2,-1)
        outs = outs.masked_fill(mask,0)
        # logging.info(outs)
        return outs, logits, ylens, filter_indice


    def calculate_all_attentions(self, xs, ilens, ys, labels, olens, spembs=None):
        """Tacotron2 forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            hs, hlens = self.enc(xs, ilens)
            if self.spk_embed_dim is not None:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
        self.train()

        return att_ws.cpu().numpy()






