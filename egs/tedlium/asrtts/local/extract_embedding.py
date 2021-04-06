#!/usr/bin/env python3
import torch
import kaldiio
import configargparse
import logging
import os
import platform
import random
import subprocess
import sys
import numpy as np
from espnet.nets.pytorch_backend.e2e_spk import SpeakerNet
from espnet.utils.deterministic_utils import set_deterministic_pytorch

def get_parser():
    parser = configargparse.ArgumentParser(
        description="extract speaker embedding",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spk-model',default=None,nargs="?",type=str,
                        help="Speaker initial model")
    parser.add_argument('--utt2spk',type=str,
                        help="utt_id to spek_id")
    parser.add_argument('--seed', default=1, type=int,
                    help='Random seed')
    parser.add_argument('--debugmode', default=1, type=int,
                    help='Debugmode')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');
    parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');
    parser.add_argument('--read-file',type=str,help="input feature file path")
    parser.add_argument('--write-file',type=str,help='output speaker embedding file path')
    return parser            


def main(cmd_args):
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')


    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_deterministic_pytorch(args)

    logging.info("total speaker is %d" % args.nClasses)
    spk_model = SpeakerNet(nClasses=args.nClasses,nPerSpeaker=args.nPerSpeaker,
                        trainfunc=args.trainfunc,nOut=512)

    if args.spk_model is not None:
        spk_model.loadParameters(args.spk_model)
    else:
        spk_model = None
    spk_model.eval()
    mean = np.array([[
        -1.7101e+08, -1.727767e+08, -1.654258e+08, -1.568423e+08, -1.47768e+08, -1.355978e+08, -1.337955e+08, -1.290715e+08, -1.292888e+08, -1.333105e+08, 
        -1.380836e+08, -1.388845e+08, -1.445241e+08, -1.438754e+08, -1.428372e+08, -1.428697e+08, -1.417773e+08, -1.400568e+08, -1.448087e+08, -1.459874e+08, 
        -1.47229e+08, -1.490556e+08, -1.499799e+08, -1.522063e+08, -1.590756e+08, -1.618226e+08, -1.651485e+08, -1.684847e+08, -1.692581e+08, -1.714363e+08, 
        -1.763494e+08, -1.776152e+08, -1.789162e+08, -1.805202e+08, -1.798933e+08, -1.818852e+08, -1.852947e+08, -1.860893e+08, -1.873477e+08, -1.889484e+08, 
        -1.873008e+08, -1.891793e+08, -1.917609e+08, -1.932594e+08, -1.934982e+08, -1.90069e+08, -1.967007e+08, -1.955583e+08, -1.932292e+08, -2.001965e+08, 
        -1.926799e+08, -2.013976e+08, -1.932717e+08, -1.997551e+08, -1.955731e+08, -1.958617e+08, -1.967825e+08, -1.952326e+08, -1.931164e+08, -1.947601e+08, 
        -1.94064e+08, -1.937533e+08, -1.93948e+08, -1.940927e+08, -1.945755e+08, -1.955468e+08, -1.96344e+08, -1.963595e+08, -1.971519e+08, -1.991344e+08, 
        -1.989762e+08, -2.000582e+08, -2.019397e+08, -2.019519e+08, -2.024301e+08, -2.031892e+08, -2.029932e+08, -2.029679e+08, -2.033156e+08, -2.033823e+08, 
        -2.03208e+08, -2.036384e+08, -2.03879e+08, -2.04647e+08, -2.06028e+08, -2.060116e+08, -2.070609e+08, -2.071168e+08, -2.083309e+08, -2.092469e+08, 
        -2.103796e+08, -2.122868e+08, -2.135678e+08, -2.144521e+08, -2.158103e+08, -2.171439e+08, -2.176665e+08, -2.191257e+08, -2.193856e+08, -2.21079e+08, 
        -2.226874e+08, -2.247855e+08, -2.267768e+08, -2.286809e+08, -2.311216e+08, -2.33142e+08, -2.352095e+08, -2.373178e+08, -2.393992e+08, -2.415607e+08, 
        -2.436022e+08, -2.450806e+08, -2.462217e+08, -2.47608e+08, -2.483978e+08, -2.495429e+08, -2.495807e+08, -2.501201e+08, -2.504308e+08, -2.506836e+08, 
        -2.518955e+08, -2.528667e+08, -2.538843e+08, -2.553601e+08, -2.571577e+08, -2.592016e+08, -2.737314e+08, -3.25694e+08 
    ]])
    var = np.array([[
        3.875797e+08, 3.972777e+08, 3.76892e+08, 3.590407e+08, 3.36797e+08, 2.982351e+08, 2.993923e+08, 2.900205e+08, 2.903182e+08, 3.00258e+08, 
        3.139445e+08, 3.133095e+08, 3.316776e+08, 3.290742e+08, 3.259625e+08, 3.292938e+08, 3.253266e+08, 3.20113e+08, 3.353506e+08, 3.40549e+08, 
        3.424283e+08, 3.454718e+08, 3.482779e+08, 3.577333e+08, 3.827005e+08, 3.899876e+08, 4.01662e+08, 4.141465e+08, 4.154033e+08, 4.238292e+08, 
        4.437099e+08, 4.463138e+08, 4.495017e+08, 4.545714e+08, 4.517053e+08, 4.601415e+08, 4.730579e+08, 4.755685e+08, 4.813327e+08, 4.884872e+08, 
        4.809006e+08, 4.883675e+08, 5.00223e+08, 5.064776e+08, 5.080264e+08, 4.91717e+08, 5.215152e+08, 5.169479e+08, 5.060737e+08, 5.381505e+08, 
        5.023963e+08, 5.430141e+08, 5.040811e+08, 5.339064e+08, 5.142676e+08, 5.158492e+08, 5.202875e+08, 5.131353e+08, 5.043084e+08, 5.129934e+08, 
        5.087678e+08, 5.064136e+08, 5.083315e+08, 5.083852e+08, 5.09834e+08, 5.150194e+08, 5.177091e+08, 5.167306e+08, 5.197394e+08, 5.282414e+08, 
        5.270312e+08, 5.324564e+08, 5.408028e+08, 5.407178e+08, 5.426285e+08, 5.456758e+08, 5.454526e+08, 5.462478e+08, 5.481372e+08, 5.508704e+08, 
        5.496423e+08, 5.518889e+08, 5.532486e+08, 5.56079e+08, 5.627578e+08, 5.617894e+08, 5.666932e+08, 5.67652e+08, 5.73079e+08, 5.768822e+08, 
        5.817027e+08, 5.912957e+08, 5.977753e+08, 6.0268e+08, 6.094717e+08, 6.166043e+08, 6.196362e+08, 6.269311e+08, 6.276106e+08, 6.369116e+08, 
        6.44361e+08, 6.551513e+08, 6.656342e+08, 6.762929e+08, 6.899264e+08, 7.008929e+08, 7.117181e+08, 7.238042e+08, 7.350025e+08, 7.47482e+08, 
        7.59422e+08, 7.681328e+08, 7.75756e+08, 7.834833e+08, 7.868992e+08, 7.938968e+08, 7.929719e+08, 7.966068e+08, 7.983973e+08, 7.993377e+08,
        8.061261e+08, 8.111478e+08, 8.169364e+08, 8.25449e+08, 8.366562e+08, 8.486715e+08, 9.377093e+08, 1.289456e+09
    ]])
    num_sum = 8.478675e+07 
    with kaldiio.ReadHelper(
        "scp:%s" % args.read_file 
    ) as reader, kaldiio.WriteHelper(
        'ark,scp:%s.ark,%s.scp' %(args.write_file,args.write_file)
    ) as writer:
        for key, numpy_array in reader:
            with torch.no_grad():
                length = len(numpy_array)
                numpy_array = numpy_array[20:-20]
                # numpy_array = numpy_array[20:]
                # numpy_array = numpy_array[:-20]
                # numpy_array = numpy_array - mean/num_sum
                # numpy_array = numpy_array / ( var/num_sum - (mean/num_sum)**2)
                torch_array = torch.from_numpy(numpy_array).unsqueeze(0).float()
                
                logging.info(torch_array.size())
                writer[key] = spk_model(torch_array).squeeze(0).numpy()
                

if __name__ == '__main__':
    main(sys.argv[1:])