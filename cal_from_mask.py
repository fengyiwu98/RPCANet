import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as F

#from utils.metrics import ROCMetric
from utils.data import *

from PIL import Image
import os
import os.path as osp
import scipy.io as scio
import numpy as np

import cv2

from utils.evaluation.roc_cruve import ROCMetric
from utils.evaluation.my_pd_fa import my_PD_FA
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.):
        
        self.base_size = base_size
        self.dataset = dataset
        if(dataset == 'NUDT-SIRST'):
            self.mat_dir = './eval/matData/NUDT-SIRST'
            self.mask_dir = './datasets/NUDT-SIRST/test/masks'
        elif(dataset == 'IRSTD-1K'):
            self.mat_dir  = './eval/matData/IRSTD-1k'
            self.mask_dir = './datasets/IRSTD-1k/test/masks'
        elif(dataset == 'SIRST-aug'):
            self.mat_dir = './eval/matData/sirst_aug'
            self.mask_dir = './datasets/sirst_aug/test/masks'
        else:
            raise NotImplementedError

        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre

        self.mat_transform = transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR)
        self.mask_transform = transforms.Resize((base_size, base_size), interpolation=Image.NEAREST)

    def __getitem__(self, i):
        name = self.file_names[i]
        mask_path = osp.join(self.mask_dir, name) + ".png"
        mat_path = osp.join(self.mat_dir, name) + ".mat"

        #print(mask_path)

        rstImg = scio.loadmat(mat_path)['T']
        rstImg = np.asarray(rstImg)


        rst_seg = np.zeros(rstImg.shape)
        rst_seg[rstImg > self.thre] = 1

        mask=cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), -1)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask /mask.max()
        
        rstImg = cv2.resize(rstImg, dsize=(self.base_size, self.base_size), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.base_size, self.base_size), interpolation = cv2.INTER_NEAREST)

        return rstImg, mask

    def __len__(self):
        return len(self.file_names)


def cal_fpr_tpr(dataname, nbins=200, fileName = None):

    f = open(fileName, mode = 'a+')
    print('Running data: {:s}'.format(dataname))
    f.write('Running data: {:s}'.format(dataname) + '\n')

    thre = 0.5

    baseSize = 256
    dataset = Dataset_mat(dataname, base_size=baseSize, thre=thre)

    roc = ROCMetric(bins=200)
    eval_PD_FA = my_PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for i in range(dataset.__len__()):
        rstImg, mask = dataset.__getitem__(i)
        size = rstImg.shape
        roc.update(pred=rstImg, label=mask)
        eval_PD_FA.update(rstImg, mask)
        eval_mIoU_P_R_F.update(labels=mask, preds=rstImg)

    fpr, tpr, auc = roc.get()
    pd, fa = eval_PD_FA.get()
    miou, prec, recall, fscore = eval_mIoU_P_R_F.get()

    print('AUC: %.6f' % (auc))
    f.write('AUC: %.6f' % (auc) + '\n')
    print('Pd: %.6f, Fa: %.8f' % (pd, fa))
    f.write('Pd: %.6f, Fa: %.8f' % (pd, fa) + '\n')
    print('mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou, prec, recall, fscore))
    f.write('mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou, prec, recall, fscore) + '\n')
    f.write('\n')

    save_dict = {'tpr': tpr, 'fpr': fpr, 'Our Pd': pd, 'Our Fa': fa}
    matDir = './eval/IndicatorResult/matResult/'
    if not os.path.exists(matDir):
        os.makedirs(matDir)
    matFile = osp.join(matDir, '{:s}.mat'.format(dataname))
    scio.savemat(matFile, save_dict)


if __name__ == '__main__':
    specific = True
    data_list = ['NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug']

    fileDir = './eval/IndicatorResult/txtResult/'
    fileName = fileDir + 'mat_result.txt'
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    f = open(fileName, mode='w+')
    f.close()
    for data in data_list:
        cal_fpr_tpr(dataname=data, nbins=200, fileName = fileName)