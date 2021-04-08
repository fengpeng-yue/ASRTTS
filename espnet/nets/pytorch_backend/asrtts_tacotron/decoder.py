import six
import logging

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet,Postnet
#from torch.cuda.amp import autocast as autocast
class ZoneOutCell(torch.nn.Module):
    """ZoneOut Cell

    This code is modified from https://github.com/eladhoffer/seq2seq.pytorch

    :param torch.nn.Module cell: pytorch recurrent cell
    :param float zoneout_prob: probability of zoneout
    """

    def __init__(self, cell, zoneout_prob=0.1):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob
        if zoneout_prob > 1.0 or zoneout_prob < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple([self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h
class Decoder(torch.nn.Module):
    """Decoder to predict the sequence of features

    This the decoder which generate the sequence of features from
    the sequence of the hidden states. The network structure is
    based on that of the tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param instance att: instance of attention class
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param function output_activation_fn: activation function for outputs
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float zoneout: zoneout rate
    :param int reduction_factor: reduction factor
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim, att,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_chans=512,
                 postnet_filts=5,
                 output_activation_fn=None,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 zoneout=0.1,
                 threshold=0.5,
                 reduction_factor=1,
                 maxlenratio=14.0,
                 minlenratio=0.0):
        super(Decoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units if prenet_layers != 0 else self.odim
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans if postnet_layers != 0 else -1
        self.postnet_filts = postnet_filts if postnet_layers != 0 else -1
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.zoneout = zoneout
        self.reduction_factor = reduction_factor
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # check attention type
        if isinstance(self.att, AttForwardTA):
            self.use_att_extra_inputs = True
        else:
            self.use_att_extra_inputs = False
        # define lstm network
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(self.dlayers):
            iunits = self.idim + self.prenet_units if layer == 0 else self.dunits
            lstm = torch.nn.LSTMCell(iunits, self.dunits)
            if zoneout > 0.0:
                lstm = ZoneOutCell(lstm, self.zoneout)
            self.lstm += [lstm]
        if prenet_layers > 0:
            self.prenet = Prenet(
                idim=odim,
                n_layers=prenet_layers,
                n_units=prenet_units,
                dropout_rate=dropout)
        else:
            self.prenet = None
        # define postnet
        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout)
        else:
            self.postnet = None
        iunits = self.idim + self.dunits if self.use_concate else self.dunits
        self.feat_out = torch.nn.Linear(iunits, self.odim * self.reduction_factor, bias=False)

        self.prob_out = torch.nn.Linear(iunits, self.reduction_factor)

    def zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.dunits)
        return init_hs

    def forward(self, hs, hlens, ys):
        """Decoder forward computation

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attention weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for _ in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        outs, logits,att_ws = [], [], []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w, prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs).view(hs.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            att_ws += [att_w]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)
        if self.reduction_factor > 1:
            before_outs = before_outs.view(before_outs.size(0), self.odim, -1)  # (B, odim, Lmax)

        after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        logits = logits

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits, att_ws
    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs
    def inference(self, h, threshold=0.5, minlenratio=0.0, maxlenratio=10.0, use_att_constraint=False,backward_window=None,forward_window=None):
        """Generate the sequence of features given the encoder hidden states

        :param torch.Tensor h: the sequence of encoder states (T, C)
        :param float threshold: threshold in inference
        :param float minlenratio: minimum length ratio in inference
        :param float maxlenratio: maximum length ratio in inference
        :return: the sequence of features (L, D)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        # setup
        assert len(h.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # setup for attention constraint
        if use_att_constraint:
            last_attended_idx = 0
        else:
            last_attended_idx = None

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs,
                    ilens,
                    z_list[0],
                    prev_att_w,
                    prev_out,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window,
                )
            else:
                att_c, att_w = self.att(
                    hs,
                    ilens,
                    z_list[0],
                    prev_att_w,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window,
                )

            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in six.moves.range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
            probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
           
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            if use_att_constraint:
                last_attended_idx = int(att_w.argmax())

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws
    #@autocast()
    def generate(self, hs, hlens, att_w_maxlen=None):
        """DECODER FORWARD CALCULATION

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param int att_w_maxlen: maximum length of att_w (only for dataparallel)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attetion weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        """
        # setup
        maxlen = int(max(hlens) * self.maxlenratio)
        each_maxlen = hlens * self.maxlenratio

        minlen = int(min(hlens) * self.minlenratio)
        logging.info("the max length of generated speech %d" % maxlen)
        # logging.info("the mix length of generated speech %d" % minlen)

        # tag for out 3000 frames
        max_frames_tag = False
        max_frames_len = 3000

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for l in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        # TODO(kan-bayashi): need to be fixed in pytorch v4
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        idx = 0
        outs,logits = [], []
        ylens = [maxlen] * hs.size(0)
        end_flag = [ False ] * hs.size(0)

        if maxlen > max_frames_len:
            logging.info("maxlen is longer than max_frames_len, some utt will may be filltered")

        while True:
            # updated index
            idx += self.reduction_factor
            # decoder calculation
            # logging.info(hs.device)
            # logging.info(hlens.device)
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w, prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            #prenet_out = self._prenet_forward(prev_out)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c], dim=1) if self.use_concate else z_list[-1]
            if self.output_activation_fn is not None:
                outs += [self.output_activation_fn(self.feat_out(zcs))]
            else:
                #one_matrix = torch.ones(self.idim + self.dunits,self.odim * self.reduction_factor).to(zcs.device)
                outs += [self.feat_out(zcs).view(hs.size(0), self.odim, -1)]
                #outs += [torch.matmul(zcs,one_matrix).view(hs.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            #probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            prev_out = outs[-1][:, :, -1].detach()
            #logging.info(torch.where(prev_out>1000))
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            #check whether to finish generation
            if idx < minlen:
                continue
            elif idx >= minlen and idx < maxlen:
                probs = torch.sigmoid(logits[-1])  # (B)
                # logging.info(self.reduction_factor)
                # logging.info(probs)
                #end_flag = True
                for i in six.moves.range(probs.size(0)):
                    # logging.info('idx=%d,%d: logit=%f, probs=%f' % (idx, i, float(logits[-1][i]), float(probs[i])))
                    if end_flag[i] == True:
                        continue
                    else:
                        if probs[i][-1] > self.threshold or idx > each_maxlen[i]:
                            end_flag[i] = True
                            ylens[i] = idx
                if end_flag == [True]*hs.size(0):
                    break

            elif idx >= maxlen:
                break
            # if idx >= max_frames_len:
            #     # # filter frame longther than 3000
            #     logging.info("this squence is longer than 3000,stop generation and filter it")
            #     max_frames_tag = True
            #     break
            #logging.info(idx)
           
        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.odim, -1)  # (B, odim, Lmax)
        #outs = outs + self._postnet_forward(outs)  # (B, odim, Lmax)
        after_outs = outs + self.postnet(outs)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)

        # filter frames longer than max frames
        indice = None
        # if max_frames_tag:
        #     indice = [i for i, x in enumerate(end_flag) if x]
        #     if len(indice):
        #         ylens = torch.IntTensor(ylens)
        #         logging.info(ylens)
        #         ylens = ylens[indice]
        #         logging.info(ylens)
        #         max_len = int(max(ylens))
        #         logits = logits[indice][:,:max_len]
        #         outs = outs[indice][:,:max_len]
        #     else:
        #         indice = [0]
        #         ylens = [max_frames_len]
        #         logits = logits[indice]
        #         outs = outs[indice]
        return after_outs, logits, ylens, indice

    def calculate_all_attentions(self, hs, hlens, ys):
        """Decoder attention calculation

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for _ in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        att_ws = []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w, prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            att_ws += [att_w]
            #prenet_out = self._prenet_forward(prev_out)
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        return att_ws
