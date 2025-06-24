import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import scipy.io as scio
import numpy as np
import cv2
from evaluation.mIoU import mIoU
from evaluation.roc_cruve import ROCMetric
from evaluation.TPFNFP import SegmentationMetricTPFNFP

_EPS = 1e-16
_TYPE = np.float32

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)

class MAE(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_mae = 0
        self.count = 0

    def step(self, pred: np.ndarray, gt: np.ndarray):
        mae = self.cal_mae(pred, gt)

        self.total_mae += mae

        self.count += 1

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> float:
        mae = np.mean(np.abs(pred - gt))
        return mae

    @property
    def average(self):
        if self.count == 0:
            return 0.0
        return self.total_mae / self.count

    def get_sum(self) -> float:
        return self.total_mae


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        self.reset()
        self.alpha = alpha

    def reset(self):
        self.sms = []
        self.total_sm = 0
        self.count = 0

    def step(self, pred: np.ndarray, gt: np.ndarray):

        sm = self.cal_sm(pred, gt)
        self.total_sm += sm
        self.count += 1

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1])
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)
        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        h, w = matrix.shape
        if matrix.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(matrix)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        h, w = gt.shape
        area = h * w
        gt_LT, gt_RT = gt[0:y, 0:x], gt[0:y, x:w]
        gt_LB, gt_RB = gt[y:h, 0:x], gt[y:h, x:w]
        pred_LT, pred_RT = pred[0:y, 0:x], pred[0:y, x:w]
        pred_LB, pred_RB = pred[y:h, 0:x], pred[y:h, x:w]
        w1, w2, w3, w4 = x * y / area, y * (w - x) / area, (h - y) * x / area, 1 - x * y / area - y * (w - x) / area - (h - y) * x / area
        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB), pred=(pred_LT, pred_RT, pred_LB, pred_RB), weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w
        x, y = np.mean(pred), np.mean(gt)
        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)
        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    @property
    def average(self):
        if self.count == 0:
            return 0.0
        return self.total_sm / self.count


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=200, thre=0.):

        self.base_size = base_size
        self.dataset = dataset

        if(dataset == 'NEU'):
            self.mat_dir = r'./pngResult/NEU/RPCANet++/mat'
            self.mask_dir = r'./datasets/NEU/test/masks'
        elif(dataset == 'Saliency'):
            self.mat_dir = r'./pngResult/Saliency/RPCANet++/mat'
            self.mask_dir = r'./datasets/Saliency/test/masks'
        else:
            raise NotImplementedError
        file_mask_names = os.listdir(self.mask_dir)
        self.file_names = [s[:-4] for s in file_mask_names]
        self.thre = thre

        self.tranform = augmentation()
        self.mat_transform = transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR)
        self.mask_transform = transforms.Resize((base_size, base_size), interpolation=Image.NEAREST)

    def __getitem__(self, i):
        name = self.file_names[i]
        mask_path = osp.join(self.mask_dir, name) + ".png"
        # mask_path = osp.join(self.mask_dir, name) + ".jpg"
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

class augmentation(object):
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


class cal_fpr_tpr():
    def __init__(self, dataname, fileName=None):
        self.fileName = fileName
        self.dataname = dataname
        self.s_measure = Smeasure()
        self.mae = MAE()
        self._reset_metrics()

    def _reset_metrics(self):
        self.s_measure = Smeasure()
        self.mae = MAE()

    def _metrics_update(self, pred, gt):
        self.s_measure.step(pred, gt)
        self.mae.step(pred, gt)

    def _metrics_ave(self):
        metrics = {}
        metrics.update({
            "S-measure": self.s_measure.average,
            "MAE": self.mae.average
        })
        return metrics

    def __getitem__(self):
        f = open(self.fileName, mode='a+')
        print('Running data: {:s}'.format(self.dataname))
        f.write('Running data: {:s}'.format(self.dataname) + '\n')

        thre = 0.5
        results = []

        baseSize = 200
        dataset = Dataset_mat(self.dataname, base_size=baseSize, thre=thre)
        roc = ROCMetric(bins=200)
        eval_mIoU = mIoU()
        eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)
        eval_mIoU_P_R_F.reset()

        for i in range(dataset.__len__()):
            rstImg, mask = dataset.__getitem__(i)
            roc.update(pred=rstImg, label=mask)
            eval_mIoU.update((torch.from_numpy(rstImg.reshape(1, 1, baseSize, baseSize)) > thre),
                             torch.from_numpy(mask.reshape(1, 1, baseSize, baseSize)))
            eval_mIoU_P_R_F.update(labels=mask, preds=rstImg)

            self._metrics_update(rstImg, mask)

            s_measure = self.s_measure.cal_sm(rstImg, mask)
            mae = self.mae.cal_mae(rstImg, mask)
            results.append({
                "S-measure": s_measure,
                "MAE": mae,
            })
        iou_our, prec, recall, fscore = eval_mIoU_P_R_F.get()

        avg_metrics = self._metrics_ave()

        results.append({
            "S-measure": avg_metrics['S-measure'],
            "MAE": avg_metrics['MAE'],
        })

        print("Averaged Metrics: ", avg_metrics)

        avg_sm = avg_metrics['S-measure']
        avg_mae = avg_metrics['MAE']

        print('Our: IoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (iou_our, prec, recall, fscore))
        f.write('Our: IoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (iou_our, prec, recall, fscore) + '\n')

        print('S-measure :%.6f MAE :%.6f' % (avg_sm, avg_mae))
        f.write('S-measure :%.6f MAE :%.6f' % (avg_sm, avg_mae) + '\n')

        f.write('\n')
        save_dict = {'sm': avg_sm, 'mae': avg_mae, 'iou': iou_our, 'f1': fscore}
        scio.savemat(osp.join('./mats', 'RPCANet++_{:s}.mat'.format(self.dataname)), save_dict)

        f.close()

if __name__ == '__main__':
    specific = True
    data_list = ['NEU']
    fileName = './txtResult/RPCANet++_NEU.txt'
    cal = cal_fpr_tpr(dataname= 'NEU', fileName = fileName)


    f = open(fileName, mode='w+')
    for data in data_list:
        cal.__getitem__()


