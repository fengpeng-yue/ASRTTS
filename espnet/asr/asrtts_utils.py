#!/usr/bin/env python

# Copyright 2017 Brno University (Karthick Baskar)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import math
# matplotlib related

# chainer related
from chainer import training

# io related
import matplotlib
import torch
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *
def merge_batchsets(train_subsets, shuffle=False):
    ndata = len(train_subsets)
    train = []
    for num in range(ndata):
        train.extend(train_subsets[num])
    if shuffle:
        import random
        random.shuffle(train)
    return train


def remove_output_layer(pretrained_model, odim, eprojs, dunits, model_type):
    stdv_bias = 1. / math.sqrt(odim)
    stdv_weight = 1. / math.sqrt(dunits)
    stdv_cweight = 1. / math.sqrt(eprojs)
    wt = torch.nn.init.uniform_(torch.FloatTensor(odim, eprojs).cuda(),
                                -stdv_cweight, stdv_cweight)
    ewt = torch.nn.init.uniform_(torch.FloatTensor(odim, dunits).cuda(),
                                 -stdv_weight, stdv_weight)
    eot = torch.nn.init.uniform_(torch.FloatTensor(odim, dunits).cuda(),
                                 -stdv_weight, stdv_weight)
    bs = torch.nn.init.uniform_(torch.FloatTensor(odim).cuda(),
                                -stdv_bias, stdv_bias)
    if model_type == 'asr':
        pretrained_model['ctc.ctc_lo.weight'] = wt
        pretrained_model['ctc.ctc_lo.bias'] = bs
        pretrained_model['dec.embed.weight'] = ewt
        pretrained_model['dec.output.weight'] = eot
        pretrained_model['dec.output.bias'] = bs
    elif model_type == 'tts':
        pretrained_model['enc.embed.weight'] = ewt
    return pretrained_model


def freeze_parameters(model, elayers, *freeze_layer):
    size = 0
    count = 0
    for child in model.children():
        logging.info(str(type(child)))
        for name, module in child.named_children():
            logging.info(str(type(module)))
            if name not in freeze_layer and name == 'enc':
                for enc_name, enc_module in module.named_children():
                    for enc_layer_name, enc_layer_module in enc_module.named_children():
                        count += 1
                        if count <= elayers:
                            logging.info(str(enc_layer_name) + " components is frozen")
                            for mname, param in module.named_parameters():
                                param.requires_grad = False
                                size += param.numel()
                        else:
                            logging.info(str(enc_layer_name) + " components is not frozen")
            elif name not in freeze_layer and name != 'enc':
                for mname, param in module.named_parameters():
                    logging.info(str(mname) + " components is frozen")
                    logging.info(str(mname) + " >> params after re-init")
                    param.requires_grad = False
                    size += param.numel()
            else:
                logging.info(str(name) + " components is not frozen")
    return model, size


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations

    Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def sgd_lr_decay(lr_decay):
    '''Extension to perform sgd lr decay'''
    @training.make_extension(trigger=(1, 'epoch'))
    def sgd_lr_decay(trainer):
        _sgd_lr_decay(trainer, lr_decay)
    return sgd_lr_decay


def _sgd_lr_decay(trainer, lr_decay):
    optimizer = trainer.updater.get_optimizer('main')
    # for chainer
    if hasattr(optimizer, 'lr'):
        current_lr = optimizer.lr
        setattr(optimizer, 'lr', current_lr * lr_decay)
        logging.info('sgd lr decayed to ' + str(optimizer.lr))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p['lr'] *= lr_decay
            logging.info('sgd lr decayed to ' + str(p["lr"]))

def torch_joint_snapshot(savefun=torch.save, filename="snapshot.ep.{.updater.epoch}"):
    """Extension to take snapshot of the trainer for pytorch.

    Returns:
        An extension function.

    """
    from chainer.training import extension

    @extension.make_extension(trigger=(1, "epoch"), priority=-100)
    def torch_joint_snapshot(trainer):
        _torch_joint_snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return torch_joint_snapshot
def _torch_joint_snapshot_object(trainer, target, filename, savefun):
    from chainer.serializers import DictionarySerializer

    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            asr_model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            asr_model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            asr_model_state_dict = trainer.updater.model.module.state_dict()
        else:
            asr_model_state_dict = trainer.updater.model.state_dict()

    if hasattr(trainer.updater.tts_model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.tts_model.model, "module"):
            tts_model_state_dict = trainer.updater.tts_model.model.module.state_dict()
        else:
            tts_model_state_dict = trainer.updater.tts_model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.tts_model, "module"):
            tts_model_state_dict = trainer.updater.tts_model.module.state_dict()
        else:
            tts_model_state_dict = trainer.updater.tts_model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "asr_model": asr_model_state_dict,
        "tts_model": tts_model_state_dict,
        "asr_optimizer": trainer.updater.get_optimizer("main").state_dict(),
        "tts_optimizer": trainer.updater.get_optimizer("tts").state_dict()
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = "tmp" + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


def torch_joint_resume(snapshot_path, trainer):
    """Resume from snapshot for pytorch.

    Args:
        snapshot_path (str): Snapshot file path.
        trainer (chainer.training.Trainer): Chainer's trainer instance.

    """
    from chainer.serializers import NpzDeserializer

    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore trainer states
    d = NpzDeserializer(snapshot_dict["trainer"])
    d.load(trainer)

    # restore asr model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict["asr_model"])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict["asr_model"])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict["asr_model"])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict["asr_model"])
    # restore tts model states
    if hasattr(trainer.updater.tts_model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.tts_model, "module"):
            trainer.updater.tts_model.module.load_state_dict(snapshot_dict["tts_model"])
        else:
            trainer.updater.tts_model.load_state_dict(snapshot_dict["tts_model"])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.tts_model, "module"):
            trainer.updater.tts_model.module.load_state_dict(snapshot_dict["tts_model"])
        else:
            trainer.updater.tts_model.load_state_dict(snapshot_dict["tts_model"])

    # retore optimizer states
    trainer.updater.get_optimizer("main").load_state_dict(snapshot_dict["asr_optimizer"])
    trainer.updater.get_optimizer("tts").load_state_dict(snapshot_dict['tts_optimizer'])
    # delete opened snapshot
    del snapshot_dict
