import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image, ImageOps, ImageFilter

import random
import os
import os.path as osp
import time
import scipy.io as scio

from models import get_model


net = get_model('rpcanet_pp')
file_path =  r'./datasets/Saliency/test/images'
pkl_file =  r'./result/DD/Saliency/RPCANet++_s6.pkl'
img_dir = r'./pngResult/Saliency/RPCANet++/img/'
mat_dir = r'./pngResult/Saliency/RPCANet++/mat/'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mat_dir, exist_ok=True)
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()
def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(42)

for filename in os.listdir(file_path):
    img = cv2.imread(file_path + '/' + filename, 0)

    w = 200
    h = 200

    crop_size = 200
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 1, w, h)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    name = os.path.splitext(filename)[0]
    matname = name+'.mat'


    with torch.no_grad():
        start = time.time()
        b, c, h, w = img.shape
        if h > crop_size and w > crop_size:
            img_unfold = F.unfold(img[:, :, :, :], crop_size, stride=crop_size)
            img_unfold = img_unfold.reshape(c, crop_size, crop_size, -1).permute(3, 0, 1, 2)
            patch_num = img_unfold.size(0)
            for pi in range(patch_num):
                img_pi = img_unfold[pi, :, :, :].unsqueeze(0).float()
                out_D_pis, out_T_pis, = net(img_pi)
                if pi == 0:
                    out_Ds, out_Ts = out_D_pis, out_T_pis
                else:
                    out_Ds, out_Ts = torch.cat([out_Ds, out_D_pis], dim=0), torch.cat([out_Ts, out_T_pis], dim=0)
            out_Ds, out_Ts = out_Ds.permute(1, 2, 3, 0).unsqueeze(0), out_Ts.permute(1, 2, 3, 0).unsqueeze(0)
            out_D, out_T = F.fold(out_Ds.reshape(1, -1, patch_num), kernel_size=crop_size, stride=crop_size,
                                  output_size=(h, w)), F.fold(out_Ts.reshape(1, -1, patch_num),
                                                              kernel_size=crop_size, stride=crop_size,
                                                              output_size=(h, w))
        else:
            out_D, out_T = net(img)
        D, T = out_D, out_T
        end = time.time()
        total = end - start
        T = F.sigmoid(T)

        D, T = D.detach().numpy().squeeze(), T.detach().numpy().squeeze()
        T[T < 0] = 0
        Tout = T * 255

        cv2.imwrite(os.path.join(img_dir, filename), Tout)
        scio.savemat(os.path.join(mat_dir, matname), {'T': T})