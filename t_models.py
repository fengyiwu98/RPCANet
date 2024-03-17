import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import os
import os.path as osp
import time

from models import get_model


net = get_model('rpcanet')
file_path =  "./datasets/NUDT-SIRST/test/images"
pkl_file = r'./result/NUDT_SIRST_mIoU_89.31.pkl'
checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()

imgDir = "./result/NUDT-SIRST/img/"
if not os.path.exists(imgDir):
    os.makedirs(imgDir)
matDir = "./result/NUDT-SIRST/mat/"
if not os.path.exists(matDir):
    os.makedirs(matDir)

for filename in os.listdir(file_path):
    img_gray = cv2.imread(file_path + '/' + filename, 0)
    img_gray = cv2.resize(img_gray, [256, 256],interpolation=cv2.INTER_LINEAR)
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
        T = T.detach().numpy().squeeze()
        T[T < 0] = 0
        Tout = T * 255

    cv2.imwrite(imgDir + filename, Tout)
    scio.savemat(matDir + matname, {'T': T})