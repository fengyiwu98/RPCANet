import torch.utils.data as Data
import random
from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import sys
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from utils.metrics import AverageMeter, get_metrics


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.):
        
        self.base_size = base_size
        self.dataset = dataset

        if(dataset == 'STARE'):
            self.mat_dir = r'./pngResult/STARE/RPCANet++/mat'
            self.mask_dir = r'./datasets/STARE/test/masks'

        elif(dataset == 'DRIVE'):
            self.mat_dir  = r'./pngResult/DRIVE/RPCANet++/mat'
            self.mask_dir = r'./datasets/DRIVE/test/masks'

        elif (dataset == 'CHASEDB1'):
            self.mat_dir = r'./pngResult/CHASEDB1/RPCANet++/mat'
            self.mask_dir = r'./datasets/CHASEDB1/test/masks'
        else:
            raise NotImplementedError

        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre


    def __getitem__(self, i):
        name = self.file_names[i]
        mask_path = osp.join(self.mask_dir, name) + ".ppm"##STARE
        # mask_path = osp.join(self.mask_dir, name) + ".png"##DRIVE
        # mask_path = osp.join(self.mask_dir, name) + ".png"##CHASEDB1
        mat_path = osp.join(self.mat_dir, name) + ".mat"

        rstImg = scio.loadmat(mat_path)['T']

        rstImg = np.asarray(rstImg)


        rst_seg = np.zeros(rstImg.shape)
        rst_seg[rstImg > self.thre] = 1

        mask = Image.open(mask_path).convert('I')

        mask = np.array(mask)


        mask = mask /mask.max()


        return rstImg, mask

    def __len__(self):
        return len(self.file_names)




class augmentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()


class MedicalEvaluation:
    def __init__(self):
        self._reset_metrics()

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):
        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "Pre": self.pre.average,
            "IOU": self.iou.average
        }

    def cal_fpr_tpr(self, dataname, nbins=200, fileName=None):
        f = open(fileName, mode='a+')
        print('Running data: {:s}'.format(dataname))
        f.write('Running data: {:s}'.format(dataname) + '\n')

        thre = 0.5

        baseSize = 256
        dataset = Dataset_mat(dataname, base_size=baseSize, thre=thre)

        for i in range(dataset.__len__()):
            rstImg, mask = dataset.__getitem__(i)
            metrics = get_metrics(rstImg, mask, thre)
            self._metrics_update(metrics['AUC'], metrics['F1'], metrics['Acc'], metrics['Sen'], metrics['Spe'],
                                 metrics['Pre'], metrics['IOU'])
            print('AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f}'.format(metrics['AUC'],
                                                                                                        metrics['F1'],
                                                                                                        metrics['Acc'],
                                                                                                        metrics['Sen'],
                                                                                                        metrics['Spe'],
                                                                                                        metrics['Pre'],
                                                                                                        metrics['IOU']))

        f.write(f'Average metrics: {self._metrics_ave()}')
        f.write('\n')
        print(f'Average metrics: {self._metrics_ave()}')




if __name__ == '__main__':
    data_list = ['STARE']

    fileName = './txtResult/STARE_RPCANet++.txt'
    f = open(fileName, mode='w+')
    f.close()

    evaluator = MedicalEvaluation()

    for data in data_list:
        evaluator.cal_fpr_tpr(dataname=data, nbins=200, fileName=fileName)


