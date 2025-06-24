from thop import profile
import torch
import os
import numpy as np
import os.path as osp
import time
import scipy.io as scio
from models import get_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
net = get_model('rpcanet_pp')
file_path =  r"./datasets/sirst_aug/test/images"
pkl_file = r"./result/ISTD/SIRST-Aug/RPCANet++_s6.pkl"
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
img =np.random.randint(255, size=(256,256),dtype=np.uint8)
img = img.reshape(1, 1, 256, 256) / 255.
img = torch.from_numpy(img).type(torch.FloatTensor)
flops, params = profile(net, inputs=(img, ))
print('Params: %2fM' % (params/1e6))
print('FLOPs: %2fGFLOPs' % (flops/1e9))
