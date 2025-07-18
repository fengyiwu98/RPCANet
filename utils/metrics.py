import threading
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import cv2

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, jaccard_score
import numpy as np
import torch.nn.functional as F

__all__ = ['SegmentationMetric', 'SegmentationMetricTPFNFP', 'ROCMetric']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_pixacc_miou(total_correct, total_label, total_inter, total_union):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return pixAcc, mIoU


def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return miou, prec, recall, fscore


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_correct, self.total_label, self.total_inter, self.total_union

    def get(self):
        return get_pixacc_miou(self.total_correct, self.total_label, self.total_inter, self.total_union)

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return


class SegmentationMetricTPFNFP(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        return get_miou_prec_recall_fscore(self.total_tp, self.total_fp, self.total_fn)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        return


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    output = output.detach().numpy()
    target = target.detach().numpy()

    predict = (output > 0).astype('int64')  # P
    pixel_labeled = np.sum(target > 0)  # T
    pixel_correct = np.sum((predict == target) * (target > 0))  # TP
    assert pixel_correct <= pixel_labeled
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    mini = 1
    maxi = nclass
    nbins = nclass

    predict = (output.detach().numpy() > 0).astype('int64')  # P
    target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all()
    return area_inter, area_union


def batch_tp_fp_fn(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    predict = (output.detach().numpy() > 0).astype('int64')  # P
    target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        # This is to compute fpr and tpr
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

        # This is to compute pd and fa
        score_thresh = 0.5
        i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)

        self.tp += i_tp
        self.pos += i_pos
        self.fp += i_fp
        self.neg += i_neg


    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        pd = self.tp / (self.pos)
        fa = self.fp / (self.neg)

        return tp_rates, fp_rates, pd, fa

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)

        self.tp = 0
        self.fp = 0
        self.pos = 0
        self.neg = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (torch.sigmoid(output).detach().numpy() > score_thresh).astype('int64') # P
    target = target.detach().numpy().astype('int64')  # T
    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    return tp, pos, fp, neg


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def average(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count






def get_metrics(predict, target, threshold=0.5):
    predict_b = (predict > threshold).astype(int)
    target = target.astype(int)

    # Flatten the arrays to ensure they are 1-dimensional
    predict_flat = predict.flatten()
    target_flat = target.flatten()

    # Calculate metrics
    auc = float('nan')  # Initialize auc with NaN
    unique_classes = np.unique(target_flat)
    # print(f"Unique classes in target: {unique_classes}")

    # Check for number of classes in the target
    if len(unique_classes) > 1:
        auc = roc_auc_score(target_flat, predict_flat)
    elif len(unique_classes) == 1 and unique_classes[0] == 1:
        # Special case where only the positive class is present in the ground truth
        auc = 1.0

    f1 = f1_score(target_flat, predict_b.flatten(), zero_division=1)
    acc = accuracy_score(target_flat, predict_b.flatten())
    sen = recall_score(target_flat, predict_b.flatten(), zero_division=1)
    spe = precision_score(target_flat, predict_b.flatten(), zero_division=1)
    pre = precision_score(target_flat, predict_b.flatten(), zero_division=1)
    iou = jaccard_score(target_flat, predict_b.flatten(), zero_division=1)

    return {
        'AUC': auc,
        'F1': f1,
        'Acc': acc,
        'Sen': sen,
        'Spe': spe,
        'Pre': pre,
        'IOU': iou
    }



def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n

