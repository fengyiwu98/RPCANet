import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import get_model
from utils.data import *
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.metrics import SegmentationMetricTPFNFP
from utils.my_pd_fa import *
from utils.pd_fa import *
from utils.logger import setup_logger
from torch.utils.data import DataLoader


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of RPCANets')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=200, help='base size of images')
    parser.add_argument('--crop-size', type=int, default=200, help='crop size of images')
    parser.add_argument('--dataset', type=str, default='neu', help='choose datasets')

    #
    # Training parameters
    #

    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler')

    #
    # Net parameters
    #
    parser.add_argument('--net-name', type=str, default='rpcanet',
                        help='net name: fcn')

    #
    # Save parameters
    #
    parser.add_argument('--save-iter-step', type=int, default=10, help='save model per step iters')
    parser.add_argument('--log-per-iter', type=int, default=10, help='interval of logging')
    parser.add_argument('--base-dir', type=str, default='./result/', help='saving dir')

    args = parser.parse_args()

    # Save folders
    #args.base_dir = r'D:\WFY\dun_irstd\result'
    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime(time.time()))
    args.folder_name = '{}_{}_{}'.format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger("Robust PCA Networks", args.save_folder, 0, filename='log.txt')
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        ## dataset

        if args.dataset == 'neu':
            trainset = NEUDatasetTrain(base_dir=r'./datasets/NEU', mode='train',
                                  base_size=args.base_size, patch_size=args.crop_size)
            valset = NEUDatasetTest(base_dir=r'./datasets/NEU', mode='test',
                                base_size=args.base_size, patch_size=args.crop_size)
        elif args.dataset == 'saliency':
            trainset = SaliencyDatasetTrain(base_dir=r'./datasets/Saliency', mode='train',
                                       base_size=args.base_size,
                                       patch_size=args.crop_size)
            valset = SaliencyDatasetTest(base_dir=r'./datasets/Saliency', mode='test',
                                    base_size=args.base_size,
                                    patch_size=args.crop_size)
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=1, shuffle=False)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        ## model
        self.net = get_model(args.net_name)
        self.net = self.net.to(self.device)

        ## criterion
        self.softiou = SoftLoULoss()
        self.mse = torch.nn.MSELoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                           args.epochs, len(self.train_data_loader), lr_step=10)

        ## optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_iou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.iou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()
        self.eval_PD_FA = PD_FA()

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, 'Args:%s, ' % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def training(self):
        # training step
        start_time = time.time()
        base_log = "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f}={:.4f}+{:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"
        for epoch in range(args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()

                self.scheduler(self.optimizer, i, epoch, self.best_iou)

                data = data.to(self.device)

                labels = labels.to(self.device)

                out_D, out_T = self.net(data)

                loss_softiou = self.softiou(out_T, labels)
                loss_mse = self.mse(out_D, data)
                gamma = torch.Tensor([0.01]).to(self.device)
                loss_all = loss_softiou + torch.mul(gamma, loss_mse)

                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/Loss All', np.mean(loss_all.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss SoftIoU', np.mean(loss_softiou.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss MSE', np.mean(loss_mse.item()), self.iter_num)
                self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(base_log.format(epoch + 1, args.epochs, self.iter_num % self.iter_per_epoch,
                                                     self.iter_per_epoch, self.optimizer.param_groups[0]['lr'],
                                                     loss_all.item(), loss_softiou.item(), loss_mse.item(),
                                                     cost_string, eta_string))

                if (self.iter_num % args.save_iter_step) == 0 or self.iter_num % self.iter_per_epoch == 0:
                    self.validation()

    def validation(self):
        self.metric.reset()
        self.net.eval()
        base_log = "Data: {:s}, IoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "


        for i, (data, labels, size) in enumerate(self.val_data_loader):
            with torch.no_grad():
                b, c, h, w = data.shape
                if h > args.crop_size and w > args.crop_size:
                    img_unfold = F.unfold(data[:, :, :, :], args.crop_size, stride=args.crop_size)
                    img_unfold = img_unfold.reshape(c, args.crop_size, args.crop_size, -1).permute(3, 0, 1, 2)
                    patch_num = img_unfold.size(0)
                    for pi in range(patch_num):
                        img_pi = img_unfold[pi, :, :, :].unsqueeze(0).float()
                        out_D_pis,  out_T_pis,= self.net(img_pi.to(self.device))
                        if pi == 0:
                            out_Ds, out_Ts = out_D_pis, out_T_pis
                        else:
                            out_Ds, out_Ts = torch.cat([out_Ds, out_D_pis], dim=0), torch.cat([out_Ts, out_T_pis], dim=0)
                    out_Ds, out_Ts = out_Ds.permute(1, 2, 3, 0).unsqueeze(0), out_Ts.permute(1, 2, 3, 0).unsqueeze(0)
                    out_D, out_T = F.fold(out_Ds.reshape(1, -1, patch_num), kernel_size=args.crop_size, stride=args.crop_size,
                                  output_size=(h, w)), F.fold(out_Ts.reshape(1, -1, patch_num), kernel_size=args.crop_size, stride=args.crop_size,
                                  output_size=(h, w))
                else:
                    out_D, out_T = self.net(data.to(self.device))
            out_D = out_D[:, :, :size[0], :size[1]]
            out_T = out_T[:, :, :size[0], :size[1]]
            labels = labels[:, :, :size[0], :size[1]]
            out_D, out_T = out_D.cpu(), out_T.cpu()


            self.metric.update(labels, out_T)


        iou, prec, recall, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'latest.pkl'))
        if iou > self.best_iou:
            self.best_iou = iou
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'best.pkl'))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure


        self.writer.add_scalar('Test/IoU', iou, self.iter_num)
        self.writer.add_scalar('Test/F1', fmeasure, self.iter_num)
        self.writer.add_scalar('Best/IoU', self.best_iou, self.iter_num)
        self.writer.add_scalar('Best/Fmeasure', self.best_fmeasure, self.iter_num)

        self.logger.info(base_log.format(self.args.dataset, iou, self.best_iou, fmeasure, self.best_fmeasure))


if __name__ == '__main__':
    args = parse_args()


    trainer = Trainer(args)
    trainer.training()

    print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_iou, trainer.best_fmeasure))
