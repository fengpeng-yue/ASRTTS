from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import numpy as np
import random
from kaldiio import ReadHelper
import argparse
import torch
import torch.nn.functional as F
parser=argparse.ArgumentParser()
parser.add_argument('feat_scp_1', help='speaker feature 1')
parser.add_argument('feat_scp_2', help='speaker feature 2')
args=parser.parse_args()
random.seed(1)
np.random.seed(1)
data = []
speaker = [] 
marker = []
alpha=[]
def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

with ReadHelper('scp:%s' % args.feat_scp_1) as reader:
    for key, numpy_array in reader:
        # torch_array = F.normalize(torch.from_numpy(numpy_array),dim=0)
        # numpy_array = torch_array.numpy()
        #print(numpy_array)
        data.append(numpy_array)
        speaker.append(key.split("_")[0])
        marker.append("x")
        alpha.append(1.0)
with ReadHelper('scp:%s' % args.feat_scp_2) as reader:
    for key, numpy_array in reader:
        # torch_array = F.normalize(torch.from_numpy(numpy_array),dim=0)
        # numpy_array = torch_array.numpy()
        data.append(numpy_array)
        speaker.append(key.split("_")[0])
        marker.append("o")
        alpha.append(0.4)


dictkeys = list(set([x for x in speaker]))
dictkeys.sort()
dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
print(dictkeys)
speaker_label = []
for item in speaker:
    speaker_label.append(dictkeys[item])
data = np.stack(data,axis=0).astype(np.float)
assert data.shape[0] == len(speaker_label)
tsne=TSNE()
data=tsne.fit_transform(data)  #进行数据降维,降成两维
print(data.shape)
key=1
r_nk=np.array(speaker_label)
if(key==1):
    #plt.scatter(x=data[:,:1].reshape(1,-1)[0],y=data[:,1:].reshape(1,-1)[0],c=r_nk)
    mscatter(x=data[:,:1].reshape(1,-1)[0],y=data[:,1:].reshape(1,-1)[0],c=r_nk,m=marker)

else:
    plt.scatter(x=data[:,:1].reshape(1,-1)[0],y=data[:,1:].reshape(1,-1)[0])
#save_path="/data/t-fyue/disk2/results/libritts_results/exp/train_clean_460_pytorch_phone_lr-decay_16k_maxframe_2000_use-guided_reduction_factor1_batch128_gpu8_ddp_new_spk/outputs_model.loss.best_decode_60/spk_embedding"
save_path="/data/t-fyue/disk2"
plt.savefig(save_path + "/move_head_specaug_speaker_before_vecoder.png")