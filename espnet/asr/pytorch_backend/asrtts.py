#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import sys

# chainer related
import chainer

#from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch
import time
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
# espnet related
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import PlotAttentionReport
from espnet.asr.asr_utils import PlotAttentionReport as TTS_PlotAttentionReport
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.asrtts_utils import torch_joint_resume
from espnet.asr.asrtts_utils import torch_joint_snapshot
from espnet.asr.asrtts_utils import merge_batchsets
from espnet.asr.asrtts_utils import remove_output_layer
from espnet.nets.pytorch_backend.e2e_asr import pad_list
# from espnet.nets.pytorch_backend.e2e_asrtts import E2E as asrtts
from espnet.nets.pytorch_backend.e2e_asr import E2E as asr
from espnet.utils.io_asrttsutils import LoadInputsAndTargetsASRTTS
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.dataset import ChainerDataLoader_joint
from espnet.utils.dataset import TransformDataset
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.nets.pytorch_backend.e2e_asrtts import Tacotron2ASRLoss
from espnet.asr.pytorch_backend.asr import CustomConverter as ASRConverter


# decode
from espnet.nets.tts_interface import TTSInterface
from espnet.nets.asr_interface import ASRInterface
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.asr.asr_utils import add_results_to_json
import time
#from espnet.utils.check_tools import spec_show
#from espnet.utils.check_grad import plot_grad_flow
# rnnlm
import espnet.lm.pytorch_backend.lm as lm_pytorch

# mixed precision
#import torch.cuda.amp.autocast as autocast
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 1
REPORT_TYPE = "epoch"
grads = {}

class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self,kl_loss, ppl, tts2asr_loss_asr, asr_tts_loss, tts2asr_acc_asr, tts2asr_acc_tts, tts2asr_loss_tts):
        reporter_module.report({'kl_loss': kl_loss}, self)
        reporter_module.report({'asr_ppl': ppl}, self)
        reporter_module.report({'tts2asr_loss_asr': tts2asr_loss_asr}, self)
        reporter_module.report({'asr2tts_loss':asr_tts_loss})
        reporter_module.report({'tts2asr_acc_asr': tts2asr_acc_asr}, self)
        reporter_module.report({'tts2asr_acc_tts': tts2asr_acc_tts}, self)
        reporter_module.report({'tts2asr_loss_tts': tts2asr_loss_tts}, self)

    def report2(self, asr_loss, asr_acc,cer,wer):
        reporter_module.report({'asr_loss': asr_loss}, self)
        reporter_module.report({'asr_acc': asr_acc}, self)
        reporter_module.report({'cer': cer}, self)
        reporter_module.report({'wer': wer}, self)

    def report3(self,tts_loss,attn_loss,mse_loss,l1_loss,bce_loss):
        reporter_module.report({'tts_loss':tts_loss},self)
        reporter_module.report({'attn_loss':attn_loss},self)
        reporter_module.report({'mse_loss':mse_loss},self)
        reporter_module.report({'l1_loss':l1_loss},self)
        reporter_module.report({'bce_loss':bce_loss},self)

class CustomEvaluator_tts(extensions.Evaluator):
    '''Custom evaluater for pytorch'''
    def __init__(self,args,tts_model, tts_iterator,target, tts_converter,device):
        super(CustomEvaluator_tts, self).__init__(tts_iterator, target)
        self.tts_model = tts_model
        self.tts_converter = tts_converter
        self.device = device
        self.args = args

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        tts_iterator = self._iterators['main']
        if self.eval_hook:
            self.eval_hook(self)
        if hasattr(tts_iterator, 'reset'):
            tts_iterator.reset()
            tts_it = tts_iterator
        else:
            tts_it = copy.copy(tts_iterator)
        summary = reporter_module.DictSummary()
        self.tts_model.eval()
        with torch.no_grad():
            for batch in tts_it:
                tts_observation = {}
                with reporter_module.report_scope(tts_observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    xs_pad, ilens, ys_pad, labels, olens, spembs= self.tts_converter(batch, self.device)
                    self.tts_model(xs_pad, ilens, ys_pad, labels, olens, spembs)
                summary.add(tts_observation)
        self.tts_model.train()
        return summary.compute_mean()    

        
class CustomEvaluator_asr(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self,args, model,asr_iterator,target, converter, device):
        super(CustomEvaluator_asr, self).__init__(asr_iterator, target)
        self.model = model
        self.converter = converter
        self.device = device
        self.args = args

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        asr_iterator = self._iterators['main']
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(asr_iterator, 'reset'):
            asr_iterator.reset()
            asr_it = asr_iterator
        else:
            asr_it = copy.copy(asr_iterator)
        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in asr_it:
                asr_observation = {}
                with reporter_module.report_scope(asr_observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    xs_pad, ilens, ys_pad= self.converter(batch, self.device)
                    self.model(xs_pad, ilens, ys_pad)
                summary.add(asr_observation)
            
        self.model.train()
        return summary.compute_mean()
class TTSConverter(object):
    def __init__(self):
        pass
    def __call__(self, batch, device=torch.device("cpu")):

        # batch should be located in list
        # logging.info(batch)
        # logging.info(len(batch))
        assert len(batch) == 1
 
        xs,ys,spembs,extras= batch[0]
        
        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)
        

        # logging.info(xs)
        # perform padding and conversion to tensor
        # eos = np.array([78])
        # xs = [ np.concatenate([x,eos]) for x in xs ]
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1 :] = 1.0

        
        # load speaker embedding
        if spembs is not None:
            spembs = torch.from_numpy(np.array(spembs)).float()
            spembs = spembs.to(device)

        return xs, ilens, ys, labels, olens, spembs

class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsampling_factor=1, use_speaker_embedding=False):
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.use_speaker_embedding = use_speaker_embedding

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        if self.use_speaker_embedding:
            try:
                spembs, ys_char_asr, ys_phone_tts = batch[0]
            except ValueError:
                logging.info("loading unpaired error")
        else:
            if train_mode == 0:
                xs, ys = batch[0]
            elif train_mode == 1:
                xs, ys, ys_phone = batch[0]


        asr_char_olens = np.array([y.shape[0] for y in ys_char_asr])
        asr_char_olens = torch.from_numpy(asr_char_olens).to(device)
            
        # perform padding and convert to tensor
        ys_pad_asr = pad_list([torch.from_numpy(y).long() for y in ys_char_asr], self.ignore_id).to(device)


        # ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        # xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)

        
        eos = np.array([78])
        ys_phone_tts = [ np.concatenate([y,eos]) for y in ys_phone_tts ]
        
        phn_olens = np.array([y.shape[0] for y in ys_phone_tts])
        phn_olens = torch.from_numpy(phn_olens).to(device)
        ys_phone_pad = pad_list([torch.from_numpy(y).long() for y in ys_phone_tts],0).to(device)
        if self.use_speaker_embedding:
            try:
                spembs = torch.from_numpy(np.array(spembs)).float().to(device)
                phn_olens,index = torch.sort(phn_olens,descending=True)
                ys_pad_asr = ys_pad_asr[index]
                ys_phone_pad = ys_phone_pad[index]
                asr_char_olens = asr_char_olens[index]
                spembs = spembs[index]
                logging.info(ys_pad_asr)
                return ys_pad_asr, ys_phone_pad, asr_char_olens, phn_olens, spembs
            except UnboundLocalError:
                return xs_pad, ilens, ys_pad,olens
        else:
            return xs_pad, ilens, ys_pad, ys_phone_pad, olens

class CustomUpdater_all(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self,train_mode,tts2asr_model, asr_model, tts_model,
                unpaired_iter,asr_paired_iter,tts_paired_iter,
                optimizer_asr, optimizer_tts,schedulers,scaler,
                converter, tts_converter,asr_converter,
                device, ngpu, use_kl, zero_att,asr2tts_policy,use_unpaired,args):

        # super(CustomUpdater_all, self).__init__({"unpaired":unpaired_iter, "tts":tts_paired_iter,"main":asr_paired_iter},
        # {"spk":optimizer_spk,"tts":optimizer_tts,"main":optimizer_asr})
        super(CustomUpdater_all, self).__init__({"asr":asr_paired_iter,"tts":tts_paired_iter,"main":unpaired_iter,},
        {"main":optimizer_asr,"tts":optimizer_tts})
        self.train_mode = train_mode
        self.tts2asr_model = tts2asr_model
        self.model = asr_model
        self.tts_model = tts_model
        self.asr_grad_clip = args.asr_grad_clip
        self.tts_grad_clip = args.tts_grad_clip
        self.scaler = scaler
        self.converter = converter
        self.tts_converter = tts_converter
        self.asr_converter = asr_converter
        self.model_device = device
        self.ngpu = ngpu
        self.use_kl = use_kl
        self.zero_att = zero_att
        self.asr2tts_policy = asr2tts_policy
        self.use_unpaired = use_unpaired
        #self.epoch = 0
        self.args = args
        self.asr_tts_update=0
        self.spk_acc = 0
        self.unpaired_augstep = 1
        self.mix_precision = args.mix_precision
        self.scheduler = schedulers
    def update_core(self):
        loss = 0.0
        # self.unpaired_augstep += 1
        asr_paired_iter = self.get_iterator('asr')
        tts_paired_iter = self.get_iterator('tts')
        unpaired_iter = self.get_iterator('main')
        optimizer_tts = self.get_optimizer('tts')
        optimizer_asr = self.get_optimizer('main')
        # Get the next batch ( a list of json files)
        epoch = unpaired_iter.epoch

        
        self.tts2asr_model.train()
        
        if self.args.update_tts2asr:
            unpaired_batch = unpaired_iter.next()
            is_new_epoch = unpaired_iter.epoch != epoch
        if self.args.update_asr:
            asr_batch = asr_paired_iter.next()
        else:
            asr_batch = None

        tts_batch = tts_paired_iter.next()
        #xs_pad, ilens, ys_pad, labels, olens, spembs= self.tts_converter(tts_batch,self.model_device)
        if self.args.mix_precision:
            optimizer_asr.zero_grad()
            optimizer_tts.zero_grad()
            self.tts2asr_model.zero_grad()
            with autocast():
                tts2asr_loss = self.tts2asr_model(asr_batch,tts_batch,unpaired_batch, iteration=self.iteration, use_spk=self.args.use_spk, model_device=self.model_device,unpaired_augstep=self.unpaired_augstep)
            self.scaler.scale(tts2asr_loss).backward()
            self.scaler.unscale_(optimizer_tts)
            self.scaler.unscale_(optimizer_asr)
        else:
            optimizer_asr.zero_grad()
            optimizer_tts.zero_grad()
            tts2asr_loss = self.tts2asr_model(asr_batch,None,unpaired_batch, iteration=self.iteration, use_spk=self.args.use_spk, model_device=self.model_device,unpaired_augstep=self.unpaired_augstep)
            tts2asr_loss.backward()
        grad_norm_tts = torch.nn.utils.clip_grad_norm_(self.tts_model.parameters(),
                                                self.tts_grad_clip)
        grad_norm_asr = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.asr_grad_clip)

        logging.info('asr:iteration={},grad norm={}'.format(self.iteration,grad_norm_asr))
        logging.info('tts:iteration={},grad norm={}'.format(self.iteration,grad_norm_tts))
        if math.isnan(grad_norm_tts) or math.isinf(grad_norm_tts):
            logging.warning('tts grad norm is nan or inf. Do not update model.')
        else:
            logging.info("tts update model")
            if self.args.mix_precision:
                self.scaler.step(optimizer_tts)
            else:
                optimizer_tts.step()
        if self.args.mix_precision:
            self.scaler.update()
        if is_new_epoch:
            self.scheduler.step()
            clr = [x['lr'] for x in optimizer_tts.param_groups]
            logging.info("the learning rate is changed to %f"% max(clr))
    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        logging.info("add iteration")
        self.unpaired_augstep += 1
        self.iteration += 1





def train(local_rank,args):
    '''Run training'''
    logging.info("false")
    if not args.use_launch:
        import torch.distributed as dist
        args.local_rank = local_rank
        rank = args.local_rank + args.node_rank * args.ngpu
        logging.info(args.local_rank)
        dist.init_process_group("nccl",
                init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                rank=rank,
                world_size=args.world_size)
    
    set_deterministic_pytorch(args)
    # if args.use_unpaired == True:
    #     torch.backends.cudnn.enabled = False
    # show pytorch version
    logging.info("pytroch version is:")
    logging.info(torch.__version__)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.asr_valid_json, 'rb') as f:
        asr_valid_json = json.load(f)['utts']
    with open(args.tts_valid_json, 'rb') as f:
        tts_valid_json = json.load(f)['utts']
    utts = list(asr_valid_json.keys())
    tts_utts = list(tts_valid_json.keys())
    idim = int(asr_valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(asr_valid_json[utts[0]]['output'][0]['shape'][1])
    tts_idim = int(tts_valid_json[tts_utts[0]]['output'][0]['shape'][1])
    tts_odim = int(tts_valid_json[tts_utts[0]]['input'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    sa_reporter = Reporter()

    # specify model architecture
    if args.asr_model_conf:
        if 'conf' in args.asr_model_conf:
            with open(args.asr_model_conf, "rb") as f:
                logging.info('reading a model config file from' + args.asr_model_conf)
                import pickle
                idim, odim, train_args = pickle.load(f)
        elif 'json' in args.asr_model_conf:
            idim, odim, train_args = get_model_conf(args.asr_model,
                                                    conf_path=args.asr_model_conf)

        asr_model = asr(idim, odim, train_args)
    else:
        asr_model = asr(idim, odim, args)

    # loading asr model
    if args.asr_model:
        if args.modify_output:
            odim = int(asr_valid_json[utts[0]]['output'][0]['shape'][1])
            if args.asr_model_conf:
                asr_model = asr(idim, odim, train_args)
            else:
                asr_model = asr(idim, odim, args)
            asr_model.load_state_dict(remove_output_layer(torch.load(args.asr_model,map_location='cpu'),
                                                          odim, args.eprojs,
                                                          args.dunits, 'asr'), strict=False)
        else:
            asr_model.load_state_dict(torch.load(args.asr_model,map_location='cpu'), strict=False)
    
    # loading tts model
    from espnet.nets.pytorch_backend.e2e_asrtts import load_tacotron_loss

    loss_fn,tts_args = load_tacotron_loss(args.tts_model_conf, args.tts_model, args, sa_reporter, \
                                train_mode=args.train_mode, asr2tts_policy=args.asr2tts_policy)


  
    # set tts_mode 
    tts_model = loss_fn.model
    if not args.update_tts:
        if args.last_tts_model:
            for n in last_tts_model.model.parameters():
                n.requires_grad = False
        for n in tts_model.parameters():
            n.requires_grad = False

    logging.info(chainer.__version__)
 
    # Setup a converter  load paired data(need speech,text and speaker embedding)
    converter = CustomConverter(1, args.use_speaker_embedding)
    # only load text and speaker embedding \
    # or only load speak (as output in data.json) and speaker embedding
    tts_converter = TTSConverter()
    # only load speech and text for evaluate
    asr_converter = ASRConverter(1)

    # tts->asr model
    tts2asr_model = Tacotron2ASRLoss(loss_fn, asr_model, args,
                                        asr_converter=asr_converter,
                                        tts_converter=tts_converter,
                                        unpaired_converter = converter,
                                        reporter=sa_reporter,
                                        weight=args.teacher_weight)



    asr_param = sum([p.nelement() for p in asr_model.parameters()])
    tts_param = sum([p.nelement() for p in tts_model.parameters()])

    if tts2asr_model is not None:
        ttsasr_param = sum([p.nelement() for p in tts2asr_model.parameters()])
        logging.info("ttsasr model total Parameters: %d ", ttsasr_param)
    
    
    logging.info("asr model total Parameters: %d ", asr_param)
    logging.info("tts model total Parameters: %d ", tts_param)

    if (args.parallel_mode == 'ddp' and args.local_rank == 0) or args.parallel_mode == 'dp':
        # write model config
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        joint_model_conf = args.outdir + '/joint_model.json'
        asr_model_conf = args.outdir + '/model.json'
        tts_model_conf = args.outdir + '/tts_model.json'
        with open(joint_model_conf, 'wb') as f:
            logging.info('writing a joint training model config file to ' + joint_model_conf)
            f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
        with open(asr_model_conf, 'wb') as f:
            logging.info('writing a asr model config file to ' + asr_model_conf)
            f.write(json.dumps((idim, odim, vars(train_args)), indent=4, sort_keys=True).encode('utf_8'))
        with open(tts_model_conf,'wb') as f:
            logging.info('writing a tts model config file to ' + tts_model_conf)
            f.write(json.dumps((tts_idim, tts_odim, vars(tts_args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = asr_model.reporter
    tts_reporter = loss_fn.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        if args.parallel_mode =="dp":
            args.batch_size *= args.ngpu
    
    # set torch device
    if args.parallel_mode == "ddp":
        if args.use_launch:
            device = torch.device("cuda",args.local_rank)
        else:
            device = torch.device("cuda",rank)
    else:
        device = torch.device("cuda:0" if args.ngpu > 0 else "cpu")
    
    tts2asr_model = tts2asr_model.to(device)
    #loss_fn = loss_fn.to(device)
    if args.parallel_mode == "ddp":
        tts2asr_model = DistributedDataParallel(tts2asr_model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
        asr_model = DistributedDataParallel(asr_model,device_ids=[args.local_rank],output_device=args.local_rank)
        if args.update_tts:
            loss_fn = DistributedDataParallel(loss_fn,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
     
    else:
        tts2asr_model = DataParallel(tts2asr_model,device_ids=[i for i in range(args.ngpu)])
        asr_model = DataParallel(asr_model,device_ids=[i for i in range(args.ngpu)])
        tts_model = DataParallel(tts_model,device_ids=[i for i in range(args.ngpu)])

    


    # Setup asr and tts optimizer
    logging.info(args.asr_opt)
    if args.asr_opt == 'adadelta':
        asr_optimizer = torch.optim.Adadelta(
            asr_model.parameters(), rho=0.95, eps=args.eps)
    elif args.asr_opt == 'adam':
        asr_optimizer = torch.optim.Adam(asr_model.parameters())
    
    if args.tts_opt == 'adadelta':
        tts_optimizer = torch.optim.Adadelta(
        tts_model.parameters(), rho=0.95, eps=args.eps)
    elif args.tts_opt == 'adam':
        tts_optimizer = torch.optim.Adam(tts_model.parameters(),lr=1e-5,eps=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(tts_optimizer, step_size=2, gamma=0.5)


    scaler = GradScaler()
    #scaler=None




    # FIXME: TOO DIRTY HACK
    
    setattr(tts_optimizer, "target", tts_reporter)
    setattr(asr_optimizer, "target", reporter)
    
    setattr(tts_optimizer, "serialize", lambda s: tts_reporter.serialize(s))
    setattr(asr_optimizer, "serialize", lambda s: reporter.serialize(s))


   


    train_json = []
    for idx in range(len(args.train_json)):
        with open(args.train_json[idx], 'rb') as f:
            train_json.append(json.load(f)['utts'])


    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)


    if args.train_mode == 0: # only use unpaired speech
        sort_key1 = "input"
        maxlen_in = 700
        maxlen_out = 150   
        #load_output = False
    elif args.train_mode == 1: # only use unpaired text
        maxlen_in = 250
        maxlen_out = 35    
        sort_key1 = "output"

    # correspond --train-json []  first path:unpaired data second data:paired data
    unpaired_set = make_batchset(train_json[0], args.unpaired_batch_size,
                                           max_length_out=maxlen_out, num_batches=args.minibatches,
                                           min_batch_size=1,
                                           shortest_first=use_sortagrad,
                                           count=args.batch_count,
                                           batch_sort_key=sort_key1,
                                           batch_bins=args.batch_bins,
                                           batch_frames_in=args.batch_frames_in,
                                           batch_frames_out=args.batch_frames_out,
                                           batch_frames_inout=args.batch_frames_inout)
    # logging.info(len(unpaired_set))
    # sys.exit()
    
    asr_paired_set = make_batchset(train_json[1], args.asr_batch_size, args.maxlen_in,
                                           args.maxlen_out, args.minibatches,
                                           min_batch_size=1,
                                           shortest_first=use_sortagrad,
                                           count=args.batch_count,
                                           batch_sort_key="input",
                                           batch_bins=args.batch_bins,
                                           batch_frames_in=args.batch_frames_in,
                                           batch_frames_out=args.batch_frames_out,
                                           batch_frames_inout=args.batch_frames_inout)
    if use_sortagrad:
        batch_sort_key = "input"
    maxlen_in=150
    maxlen_out=400
    tts_paired_set = make_batchset(
        train_json[2],
        args.tts_batch_size,
        maxlen_in,
        maxlen_out,
        args.minibatches,
        batch_sort_key="output",
        min_batch_size=1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        swap_io=True,
        iaxis=0,
        oaxis=0,
    )
 
    asr_valid = make_batchset(asr_valid_json, args.asr_batch_size, args.maxlen_in, args.maxlen_out,
                            args.minibatches, min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                            count=args.batch_count,
                            batch_bins=args.batch_bins,
                            batch_frames_in=args.batch_frames_in,
                            batch_frames_out=args.batch_frames_out,
                            batch_frames_inout=args.batch_frames_inout)
    tts_valid = make_batchset(
        tts_valid_json,
        args.tts_batch_size,
        maxlen_in,
        maxlen_out,
        args.minibatches,
        batch_sort_key="input",
        min_batch_size=1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        swap_io=True,
        iaxis=0,
        oaxis=0,
    )


    load_tr = LoadInputsAndTargetsASRTTS(
        mode='asr', use_speaker_embedding=args.use_speaker_embedding,
        load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True},  # Switch the mode of preprocessing
    )
    load_tts = LoadInputsAndTargets(
        mode="tts",
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=False,
        preprocess_conf=None,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        keep_all_data_on_mem=False,
    )
    load_cv_asr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    load_cv_tts = LoadInputsAndTargets(
        mode="tts",
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=False,
        preprocess_conf=None,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
        keep_all_data_on_mem=False
    )
    # hack to make batchsze argument as 1
    # actual batchsize is included in a list
    
    #use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if args.parallel_mode == 'ddp':
        from espnet.utils.dataset import TransformDataset
        unpaired_dataset = TransformDataset(unpaired_set, lambda data: [load_tr(data,data_aug=False,speaker_mode=False)])
        unpaired_sampler = torch.utils.data.distributed.DistributedSampler(unpaired_dataset)
        if not args.use_launch:
            args.n_iter_processes = 0
        unpaired_iter = ChainerDataLoader_joint(
        name="unpaired_data",dataset=unpaired_dataset,batch_size=1,collate_fn=lambda x: x[0],num_workers=args.n_iter_processes,sampler=unpaired_sampler,parallel_mode=args.parallel_mode)


        asr_paired_dataset = TransformDataset(asr_paired_set, lambda data: [load_tr(data)])
        asr_paired_sampler = torch.utils.data.distributed.DistributedSampler(asr_paired_dataset)
        asr_paired_iter = ChainerDataLoader_joint(
        name="asr_paired_data",dataset=asr_paired_dataset,batch_size=1,collate_fn=lambda x: x[0],num_workers=args.n_iter_processes,sampler=asr_paired_sampler,parallel_mode=args.parallel_mode)


        tts_paired_dataset = TransformDataset(tts_paired_set, lambda data: [load_tts(data)])
        tts_paired_sampler = torch.utils.data.distributed.DistributedSampler(tts_paired_dataset)
        tts_paired_iter = ChainerDataLoader_joint(
        name="tts_paired_data",dataset=tts_paired_dataset,batch_size=1,collate_fn=lambda x: x[0],num_workers=args.n_iter_processes,sampler=tts_paired_sampler,parallel_mode=args.parallel_mode)



        asr_valid_iter = {'main': ChainerDataLoader_joint(
                    name="asr_valid_data",dataset=TransformDataset(asr_valid, lambda data: [load_cv_asr(data)]),
                    batch_size=1, collate_fn=lambda x: x[0],
                    num_workers=args.n_iter_processes)}
        tts_valid_iter = {'main': ChainerDataLoader_joint(
                    name="tts_valid_data",dataset=TransformDataset(tts_valid, lambda data: [load_cv_tts(data)]),
                    batch_size=1, collate_fn=lambda x: x[0],
                    num_workers=args.n_iter_processes)}
    elif args.parallel_mode == 'dp':
        # train_iter = {'main': ChainerDataLoader(
        # dataset=train_dataset,batch_size=1,collate_fn=lambda x: x[0])}
        from chainer.datasets import TransformDataset
        if args.n_iter_processes > 0:
            train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
            valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        else:
            train_iter = ToggleableShufflingSerialIterator(
                TransformDataset(train, load_tr),
                batch_size=1, shuffle=not use_sortagrad)
            valid_iter = ToggleableShufflingSerialIterator(
                TransformDataset(valid, load_cv),
                batch_size=1, repeat=False, shuffle=False)
    else:
        raise NotImplementedError

  
    #spk_optimizer=None
    # Set up a trainer
    updater = CustomUpdater_all(
        args.train_mode,
        tts2asr_model,
        asr_model,
        loss_fn,
        unpaired_iter,
        asr_paired_iter,
        tts_paired_iter,
        asr_optimizer,
        tts_optimizer,
        scheduler,
        scaler,
        converter,
        tts_converter,
        asr_converter,
        device,
        args.ngpu,
        args.use_kl,
        zero_att=args.zero_att,
        asr2tts_policy=args.asr2tts_policy,
        use_unpaired=args.use_unpaired,
        args=args)

    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)
    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_joint_resume(args.resume, trainer)
    # Evaluate the model with the test dataset for each epoch
    if args.update_asr and  not args.update_tts:
        log_acc = "validation/main/acc"
        log_loss = "validation/main/loss_att"
    else:
        log_acc ="validation_1/main/acc"
        log_loss = "validation_1/main/loss_att"
        
    if args.update_tts:
        trainer.extend(CustomEvaluator_tts(args,loss_fn,tts_valid_iter, tts_reporter,tts_converter,device),trigger=(REPORT_INTERVAL, REPORT_TYPE))
    if args.update_asr:
        trainer.extend(CustomEvaluator_asr(args,asr_model, asr_valid_iter, reporter, asr_converter,device),trigger=(REPORT_INTERVAL, REPORT_TYPE))
    
    if args.parallel_mode == "ddp":
        if args.local_rank==0:
            # logging.info(args.asr_valid_json)
            # logging.info(asr_valid_json)
            log_manager(args,asr_model,tts_model,asr_converter,tts_converter,device,load_cv_asr,load_cv_tts,trainer,asr_valid_json,tts_valid_json,mtl_mode,log_loss,log_acc)
    else:
        log_manager(args,asr_model,tts_model,asr_converter,tts_converter,device,load_cv_asr,load_cv_tts,trainer,asr_valid_json,tts_valid_json,mtl_mode)

    
    # epsilon decay in the optimizer
    if args.update_asr:
        if args.opt == 'adadelta':
            if args.criterion == 'acc' and mtl_mode != 'ctc':
                trainer.extend(restore_snapshot(asr_model, args.outdir + '/model.acc.best', load_fn=torch_load),
                            trigger=CompareValueTrigger(
                                log_acc,
                                lambda best_value, current_value: best_value > current_value
                                ,trigger=(REPORT_INTERVAL, REPORT_TYPE))
                                )
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                            trigger=CompareValueTrigger(
                                log_acc,
                                lambda best_value, current_value: best_value > current_value
                                ,trigger=(REPORT_INTERVAL, REPORT_TYPE))
                                )
            elif args.criterion == 'loss':
                trainer.extend(restore_snapshot(asr_model, args.outdir + '/model.loss.best', load_fn=torch_load),
                            trigger=CompareValueTrigger(
                                log_loss,
                                lambda best_value, current_value: best_value < current_value
                                ,trigger=(REPORT_INTERVAL, REPORT_TYPE))
                                )
                trainer.extend(adadelta_eps_decay(args.eps_decay),
                            trigger=CompareValueTrigger(
                                log_loss,
                                lambda best_value, current_value: best_value < current_value
                                ,trigger=(REPORT_INTERVAL, REPORT_TYPE))
                                )
    # Run the training
    trainer.run()

def log_manager(args,asr_model,tts_model,asr_converter,tts_converter,device,load_cv_asr,load_cv_tts,trainer,asr_valid_json,tts_valid_json,mtl_mode,log_loss,log_acc):
     
    #logging.info(type(trainer))
    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        asr_data = sorted(list(asr_valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)

        if hasattr(asr_model, "module"):
            att_vis_fn = asr_model.module.calculate_all_attentions
        else:
            att_vis_fn = asr_model.calculate_all_attentions
        trainer.extend(PlotAttentionReport(
            att_vis_fn, asr_data, args.outdir + "/att_ws_asr",
            converter=asr_converter, transform=load_cv_asr, device=device),
            trigger=(REPORT_INTERVAL, REPORT_TYPE))
        



        tts_data = sorted(
            list(tts_valid_json.items())[: args.num_save_attention],
            key=lambda x: int(x[1]["output"][0]["shape"][0]),
        )
        if hasattr(tts_model, "module"):
            att_vis_fn = tts_model.module.calculate_all_attentions
            plot_class = TTS_PlotAttentionReport
            reduction_factor = tts_model.module.reduction_factor
        else:
            att_vis_fn = tts_model.calculate_all_attentions
            plot_class = TTS_PlotAttentionReport
            reduction_factor = tts_model.reduction_factor

        if reduction_factor > 1:
            # fix the length to crop attention weight plot correctly
            tts_data = copy.deepcopy(tts_data)
            for idx in range(len(tts_data)):
                ilen = tts_data[idx][1]["input"][0]["shape"][0]
                tts_data[idx][1]["input"][0]["shape"][0] = ilen // reduction_factor
        att_reporter = plot_class(
            att_vis_fn,
            tts_data,
            args.outdir + "/att_ws_tts",
            converter=tts_converter,
            transform=load_cv_tts,
            device=device,
            reverse=True,
        )
        trainer.extend(att_reporter, trigger=(REPORT_INTERVAL, REPORT_TYPE))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'tts/kl_loss','tts2asr_loss_asr','asr2tts_loss',
                                          log_loss,'validation/tts/kl_loss'],
                                         'epoch', file_name='loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/kl_loss','validation/tts/kl_loss'],
                                         'epoch', file_name='KL_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['main/acc','tts/tts2asr_acc_asr',
                                          log_acc],
                                         'epoch', file_name='acc.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['main/loss', 
                                          log_loss],
                                         'epoch', file_name='asr_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/tts2asr_loss_asr'],
                                         'epoch', file_name='tts2asr_loss_asr.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/tts2asr_acc_asr'],
                                         'epoch', file_name='tts2asr_acc_asr.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    
    trainer.extend(extensions.PlotReport(['asr2tts_loss'],
                                        'epoch', file_name='asr2tts_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/tts_loss',
                                        'validation/main/tts_loss'],
                                        'epoch', file_name='tts_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/attn_loss',
                                        'validation/main/attn_loss'],
                                        'epoch', file_name='attn_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['main/mse_loss',
                                        'validation/main/mse_loss'],
                                        'epoch', file_name='mse_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['main/l1_loss',
                                        'validation/main/bce_loss'],
                                        'epoch', file_name='l1_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/bce_loss',
                                         'validation/main/bce_loss'],
                                        'epoch', file_name='bce_loss.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
        
    
    trainer.extend(extensions.PlotReport(['tts/tts2asr_acc_asr'],
                                        'epoch', file_name='tts2asr_acc_asr.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))
    trainer.extend(extensions.PlotReport(['tts/tts2asr_loss_asr'],
                                        'epoch', file_name='tts2asr_loss_asr.png',trigger=(REPORT_INTERVAL, REPORT_TYPE)))



    # Save best models
    if args.update_asr:
        trainer.extend(
            snapshot_object(asr_model, "asr_model.loss.best"),
            trigger=training.triggers.MinValueTrigger(log_loss,trigger=(REPORT_INTERVAL, REPORT_TYPE))
            )
        if mtl_mode != 'ctc':
            trainer.extend(
            snapshot_object(asr_model, 'model.acc.best'),
                        trigger=training.triggers.MaxValueTrigger(log_acc,trigger=(REPORT_INTERVAL, REPORT_TYPE))
                        )
    if args.update_tts:
        trainer.extend(
            snapshot_object(tts_model, "tts_model.loss.best"),
            trigger=training.triggers.MinValueTrigger("tts/tts_loss",trigger=(REPORT_INTERVAL, REPORT_TYPE))
            )
    #trainer.extend(extensions.snapshot_object(asr_model, 'model.loss.best', savefun=torch_save),
    #               trigger=training.triggers.MinValueTrigger('validation/main/asr_loss_att',trigger=(REPORT_INTERVAL, 'iteration')))
 
        

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_joint_snapshot(),trigger=(REPORT_INTERVAL, REPORT_TYPE))
    #trainer.extend(torch_snapshot(),trigger=(REPORT_INTERVAL, 'iteration'))

    

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 
                   'main/tts2asr_loss_asr','main/asr2tts_loss','main/asr_acc', 'main/tts2asr_acc_asr',
                   'main/tts2asr_loss_tts','main/tts2asr_acc_tts',
                   'validation/main/loss','validation/main/asr_acc', 
                   'main/tts_loss','main/att_loss','main/mse_loss','main/l1_loss','main/bce_loss'
                   'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(100, 'iteration'))
        report_keys.append('eps')
    if args.report_cer:
        report_keys.append('validation/main/cer')
    if args.report_wer:
        report_keys.append('validation/main/wer')
    # trainer.extend(extensions.PrintReport(
    #     report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    #trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(
            TensorboardLogger(
                SummaryWriter(args.tensorboard_dir),
                att_reporter=None,
                ctc_reporter=None,
            ),
            trigger=(REPORT_INTERVAL, "iteration"),
        )


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # load asr model
    logging.info("###########loading asr model##########")
    asr_model, asr_train_args = load_trained_model(args.asr_model)
    logging.info(asr_train_args.model_module)
    assert isinstance(asr_model, ASRInterface)
    asr_model.recog_args = args
    logging.info("##########loading asr model end#######")


    # load tts model
    logging.info("###########loading tts model##########")
    # read training config
    tts_idim, tts_odim,tts_train_args = get_model_conf(args.tts_model, args.tts_model_conf)

    # define model
    logging.info(tts_train_args.model_module)
    tts_model_class = dynamic_import(tts_train_args.model_module)
    tts_model = tts_model_class(tts_idim, tts_odim, tts_train_args)
    #assert isinstance(tts_model, TTSInterface)

    # load trained model parameters
    logging.info("reading model parameters from " + args.tts_model)
    torch_load(args.tts_model, tts_model)
    logging.info("##########loading tts model end#######")
    logging.info(tts_model)
    logging.info(asr_model)

    asr_model.eval()
    tts_model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    asr_model = asr_model.to(device)
    tts_model = tts_model.to(device)
    
    if args.streaming_mode and "transformer" in asr_train_args.model_module:
        raise NotImplementedError("streaming mode for transformer is not implemented")
    logging.info(
        " Total asr parameter of the model = "
        + str(sum(p.numel() for p in asr_model.parameters()))
    )
    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info("args: " + key + ": " + str(vars(args)[key]))
    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError(
                "use '--api v2' option to decode with non-default language model"
            )
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor, rnnlm.predictor, word_dict, char_dict
                )
            )
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor, word_dict, char_dict
                )
            )

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargetsASRTTS(
        mode='asr', use_speaker_embedding=args.use_speaker_embedding,
        load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    tts_decode_args = copy.deepcopy(args)
    tts_decode_args.minlenratio = tts_decode_args.tts_minlenratio
    tts_decode_args.maxlenratio = tts_decode_args.tts_maxlenratio

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                data = load_inputs_and_targets(batch)
                speech,spek,char_text,phone_text = data
                # logging.info(char_text)
                # logging.info(phone_text)
                
                #add eos to sentence
                eos = np.array([78])
                phone_text = np.concatenate([phone_text[0],eos])
                x = torch.LongTensor(phone_text).to(device)
                spemb = None
                if tts_train_args.use_speaker_embedding:
                    spemb = torch.FloatTensor(spek)[0].to(device)
                # logging.info(spemb)

                #logging.info(char_text.size()
                # decode and write
                
                start_time = time.time()
                
                feat, probs, att_ws = tts_model.inference(x, tts_decode_args, spemb=spemb)

                if args.streaming_mode == "window" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with window size %d frames",
                        args.streaming_window,
                    )
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info(
                            "Feeding frames %d - %d", i, i + args.streaming_window
                        )
                        se2e.accept_input(feat[i : i + args.streaming_window])
                    logging.info("Running offline attention decoder")
                    se2e.decode_with_attention_offline()
                    logging.info("Offline attention decoder finished")
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    logging.info(
                        "Using streaming recognizer with threshold value %d",
                        args.streaming_min_blank_dur,
                    )
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                elif hasattr(asr_model, "decoder_mode") and asr_model.decoder_mode == "maskctc":
                    nbest_hyps = model.recognize_maskctc(
                        feat, args, train_args.char_list
                    )
                else:
                    nbest_hyps = asr_model.recognize(
                        feat, args, asr_train_args.char_list, rnnlm
                    )
                new_js[name] = add_results_to_json(
                    js[name], nbest_hyps, asr_train_args.char_list
                )

    else:

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]["input"][0]["shape"][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = (
                    load_inputs_and_targets(batch)[0]
                    if args.num_encs == 1
                    else load_inputs_and_targets(batch)
                )
                if args.streaming_mode == "window" and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == "segment" and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({"yseq": [], "score": 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i : i + r])
                        if hyps is not None:
                            text = "".join(
                                [
                                    train_args.char_list[int(x)]
                                    for x in hyps[0]["yseq"][1:-1]
                                    if int(x) != -1
                                ]
                            )
                            text = text.replace(
                                "\u2581", " "
                            ).strip()  # for SentencePiece
                            text = text.replace(model.space, " ")
                            text = text.replace(model.blank, "")
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]["yseq"].extend(hyps[n]["yseq"])
                                nbest_hyps[n]["score"] += hyps[n]["score"]
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(
                        feats, args, train_args.char_list, rnnlm=rnnlm
                    )

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyp, train_args.char_list
                    )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
