
import six

import torch
import logging

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence



class Encoder(torch.nn.Module):
    """Character embedding encoder

    This is the encoder which converts the sequence of characters into
    the sequence of hidden states. The network structure is based on
    that of tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param bool use_batch_norm: whether to use batch normalization
    :param float dropout: dropout rate
    """

    def __init__(self, idim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_chans=512,
                 econv_filts=5,
                 use_batch_norm=True,
                 use_residual=False,
                 dropout=0.5,
                 train_mode=None,
                 asr2tts_policy=None):
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_chans = econv_chans if econv_layers != 0 else -1
        self.econv_filts = econv_filts if econv_layers != 0 else -1
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout = dropout
        self.train_mode = train_mode
        self.asr2tts_policy = asr2tts_policy
        # define network layer modules
        self.embed = torch.nn.Embedding(self.idim, self.embed_dim)
        if self.econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(self.econv_layers):
                ichans = self.embed_dim if layer == 0 else self.econv_chans
                if self.use_batch_norm:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(self.econv_chans),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
        else:
            self.convs = None
        iunits = econv_chans if self.econv_layers != 0 else self.embed_dim
        self.blstm = torch.nn.LSTM(
            iunits, self.eunits // 2, self.elayers,
            batch_first=True,
            bidirectional=True)

    def forward(self, xs, ilens, softargmax=False,asr2tts=False):
        """Character encoding forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)
        :return: batch of sequences of padded encoder states (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: batch of lengths of each encoder states (B)
        :rtype: list
        """
        # logging.info(self.asr2tts_policy)
        # logging.info(asr2tts)
        if self.train_mode == 1:
            xs = self.embed(xs).transpose(1, 2)
        
        else:
            if self.asr2tts_policy == "straight_through" and asr2tts:
                xs = torch.matmul(xs,self.embed.weight).transpose(1, 2)
            else:
                xs = self.embed(xs).transpose(1, 2)
        for l in six.moves.range(self.econv_layers):
            if self.use_residual:
                xs += self.convs[l](xs)
            else:
                xs = self.convs[l](xs)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens, batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x):
        """Character encoder inference

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence encoder states (T, eunits)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0][0]