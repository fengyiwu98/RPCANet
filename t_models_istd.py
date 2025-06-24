import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import os.path as osp
import time
import scipy.io as scio
import numpy as np

from models import get_model

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img

net = get_model('rpcanet_pp')
pkl_file = r'./result/ISTD/SIRST-Aug/RPCANet++_s6.pkl'
file_path = r'./datasets/sirst-aug/test/images'
img_dir = r'./pngResult/SIRST-Aug/RPCANet++/img/'
mat_dir = r'./pngResult/SIRST-Aug/RPCANet++/mat/'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mat_dir, exist_ok=True)
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()

for filename in os.listdir(file_path):
    img_gray = cv2.imread(file_path + '/' + filename, 0)
    img_gray = cv2.resize(img_gray, [256, 256], interpolation=cv2.INTER_LINEAR)
    img = img_gray.reshape(1, 1, 256, 256) / 255.
    img = torch.from_numpy(img).type(torch.FloatTensor)
    name = os.path.splitext(filename)[0]
    matname = name+'.mat'

    with torch.no_grad():
        start = time.time()
        D, T = net(img)
        end = time.time()
        total = end - start
        T = F.sigmoid(T)

        D, T = D.detach().numpy().squeeze(), T.detach().numpy().squeeze()
        T[T < 0] = 0
        Tout = T * 255
    cv2.imwrite(os.path.join(img_dir, filename), Tout)
    scio.savemat(os.path.join(mat_dir, matname), {'T': T})
