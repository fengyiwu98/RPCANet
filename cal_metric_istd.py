import torch
import torch.utils.data as Data


import os
import os.path as osp
import scipy.io as scio
import numpy as np


import cv2

from evaluation.mIoU import mIoU
from evaluation.roc_cruve import ROCMetric
from evaluation.pd_fa import PD_FA
from evaluation.my_pd_fa import my_PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.):
        
        self.base_size = base_size
        self.dataset = dataset
        if(dataset == 'NUDT-SIRST'):
            self.mat_dir = r'./pngResult/NUDT-SIRST/RPCANet++/mat'
            self.mask_dir = r'./datasets/NUDT-SIRST/test/masks'
        elif(dataset == 'IRSTD-1K'):
            self.mat_dir = r'./pngResult/IRSTD-1k/RPCANet++/mat'
            self.mask_dir = r'./datasets/IRSTD-1k/test/masks'
        elif (dataset == 'SIRST'):
            self.mat_dir = r'./pngResult/SIRST/RPCANet++/mat'
            self.mask_dir = r'./datasets/sirst/test/masks'
        elif(dataset == 'SIRST-aug'):
            self.mat_dir = r'./pngResult/SIRST-aug/RPCANet++/mat'
            self.mask_dir = r'./datasets/sirst_aug/test/masks'
        else:
            raise NotImplementedError

        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre

    def __getitem__(self, i):
        name = self.file_names[i]
        mask_path = osp.join(self.mask_dir, name) + ".png"
        mat_path = osp.join(self.mat_dir, name) + ".mat"

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
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    eval_my_PD_FA = my_PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for i in range(dataset.__len__()):
        rstImg, mask = dataset.__getitem__(i)
        size = rstImg.shape
        roc.update(pred=rstImg, label=mask)
        eval_mIoU.update((torch.from_numpy(rstImg.reshape(1,1,baseSize, baseSize))>thre), torch.from_numpy(mask.reshape(1,1,baseSize, baseSize)))
        eval_PD_FA.update(rstImg, mask, size)
        eval_my_PD_FA.update(rstImg, mask)
        eval_mIoU_P_R_F.update(labels=mask, preds=rstImg)

    fpr, tpr, auc = roc.get()
    pd_our, fa_our = eval_my_PD_FA.get()
    miou_our, prec, recall, fscore = eval_mIoU_P_R_F.get()

    print('AUC: %.6f' % (auc))
    f.write('AUC: %.6f' % (auc) + '\n')
    print('Our: IoU: %.6f, fscore: %.6f, Prec: %.6f, Recall: %.6f, Pd: %.6f, Fa: %.7f'% (miou_our, fscore, prec, recall,pd_our, fa_our))
    f.write('Our: IoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou_our, prec, recall, fscore) + '\n')
    f.write('\n')

    save_dict = {'tpr': tpr, 'fpr': fpr, 'Our Pd': pd_our, 'Our Fa': fa_our}
    scio.savemat(osp.join('./mats', 'RPCANet++_SIRST-Aug_{:s}.mat'.format(dataname)), save_dict)


if __name__ == '__main__':
    specific = True
    data_list = ['SIRST-Aug']

    fileName = r'./txtResult/RPCANet++_SIRST-Aug.txt'
    f = open(fileName, mode='w+')
    for data in data_list:
        cal_fpr_tpr(dataname=data, nbins=200, fileName = fileName)

