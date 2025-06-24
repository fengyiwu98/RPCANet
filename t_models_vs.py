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


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'CHASEDB1':
        img_norm_cfg = {'mean': 59.87125015258789, 'std': 46.84417724609375}
    elif dataset_name == 'DRIVE':
        img_norm_cfg = {'mean': 83.59488677978516, 'std': 54.175140380859375}
    elif dataset_name == 'STARE':
        img_norm_cfg = {'mean': 98.16558837890625, 'std': 52.33002853393555}

    else:
        with open(dataset_dir + '/' + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        if os.path.exists(dataset_dir + '/' +  '/img_idx/test_' + dataset_name + '.txt'):
            with open(dataset_dir + '/' +  '/img_idx/test_' + dataset_name + '.txt', 'r') as f:
                test_list = f.read().splitlines()
        else:
            test_list = []
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' +  '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    try:
                        img = Image.open((img_dir + img_pth).replace('//', '/') + '.pgm').convert('I')
                    except:
                        img = Image.open((img_dir + img_pth).replace('//', '/') + '.ppm').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
        print(dataset_name + '\t' + str(img_norm_cfg))
    return img_norm_cfg


img_norm_cfg = get_img_norm_cfg('STARE', './datasets/STARE')
net = get_model('rpcanet_pp')
file_path =  "./datasets/STARE/test/images"
pkl_file = r"./result/VS/STARE/best.pkl"
img_dir = "./pngResult/STARE/RPCANet++/img/"
mat_dir = "./pngResult/STARE/RPCANet++/mat/"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mat_dir, exist_ok=True)
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()


for filename in os.listdir(file_path):
    img = Image.open(file_path + '/' + filename).convert('I')
    w = 700 #STARE
    h = 605 #STARE
    # w = 565 #DRIVE
    # h = 584 #DRIVE
    # w = 999 #CHASEDB1
    # h = 960 #CHASEDB1

    img = Normalized(np.array(img, dtype=np.float32), img_norm_cfg)
    img = PadImg(img)
    print(img.shape)
    img= img[np.newaxis, :]
    img = torch.from_numpy(np.ascontiguousarray(img))
    img = img.unsqueeze(0)
    name = os.path.splitext(filename)[0]
    matname = name+'.mat'

    with torch.no_grad():
        start = time.time()
        D, T = net(img)
        T = T[:, :, :h, :w]
        D = D[:, :, :h, :w]
        end = time.time()
        total = end - start
        T = F.sigmoid(T)

        D, T = D.detach().numpy().squeeze(), T.detach().numpy().squeeze()
        T[T < 0] = 0
        Tout = T * 255

    img_pil = Image.fromarray((T * 255).astype(np.uint8)).convert('RGB')
    img_pil.save(os.path.join(img_dir, filename))
    scio.savemat(os.path.join(mat_dir, matname), {'T': T})


