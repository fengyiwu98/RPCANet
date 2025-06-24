import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import os
import os.path as osp
import sys
import random
import scipy.io as scio
import numpy as np

__all__ = ['SirstDataset', 'SirstAugDataset', 'IRSTD1kDataset', 'NUDTDataset',
           'DriveDatasetTrain', 'DriveDatasetTest','CHASEDB1DatasetTrain', 'CHASEDB1DatasetTest','STAREDatasetTrain', 'STAREDatasetTest',
           'NEUDatasetTrain', 'NEUDatasetTest', 'SaliencyDatasetTrain', 'SaliencyDatasetTest']

class NUDTDataset(Data.Dataset):
    '''
    Return: Single channel
    '''

    def __init__(self, base_dir=r'D:/WFY/datasets/NUDT',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)

class IRSTD1kDataset(Data.Dataset):
    '''
    Return: Single channel
    '''

    def __init__(self, base_dir=r'D:/WFY/datasets/IRSTD-1k',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        row, col = img.shape
        img = img.reshape(1, row, col) / 255.


        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)

class SirstDataset(Data.Dataset):
    def __init__(self, base_dir=r'D:\WFY\datasets\sirst',
                 mode='train', base_size=256):
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test1.txt'

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.base_size = base_size
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)


        if self.mode == 'train':
            img, mask = self.tranform(img, mask)
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
            img = img.reshape(1, self.base_size, self.base_size) / 255.
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            return img, mask

        elif self.mode == 'val':
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
            img = img.reshape(1, self.base_size, self.base_size) / 255.
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
            _, h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            return img, mask



    def __len__(self):
        return len(self.names)

class SirstAugDataset(Data.Dataset):
    '''
    Return: Single channel
    '''
    def __init__(self, base_dir=r'/Users/tianfangzhang/Program/DATASETS/sirst_aug',
                 mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img, mask = self.tranform(img, mask)
        img = img.reshape(1, self.base_size, self.base_size) / 255.
        if np.max(mask) > 0:
            mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
        else:
            mask = mask.reshape(1, self.base_size, self.base_size)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        return img, mask

    def __len__(self):
        return len(self.names)

class DriveDatasetTrain(Data.Dataset):
    '''
    Return: Single channel
    '''
    def __init__(self, base_dir=r'../datasets/DRIVE',
                 mode='train', base_size=512, patch_size = 256, img_norm_cfg = None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError


        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('DRIVE', '../datasets/DRIVE')
        else:
            self.img_norm_cfg = img_norm_cfg

        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('I')
        mask = Image.open(label_path).convert('I')  # Convert to grayscale
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.names)

class DriveDatasetTest(Data.Dataset):
    '''
    Return: Single channel
    '''
    def __init__(self, base_dir=r'../datasets/DRIVE',
                 mode='train', base_size=584, img_norm_cfg = None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('DRIVE', '../datasets/DRIVE')
        else:
            self.img_norm_cfg = img_norm_cfg

        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('I')
        mask = Image.open(label_path).convert('I')  # Convert to grayscale
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask

    def __len__(self):
        return len(self.names)

class STAREDatasetTrain(Data.Dataset):

    def __init__(self, base_dir=r'../datasets/STARE',
                 mode='train', base_size=700, patch_size=256, img_norm_cfg=None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('STARE', r'../datasets/STARE')
        else:
            self.img_norm_cfg = img_norm_cfg
        self.base_size = base_size
        self.patch_size = patch_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('ppm'):
                self.names.append(filename)
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('I')  # It also can take use of .ppm file
        mask = Image.open(label_path).convert('I')  # Convert to grayscale
        img = np.array(img, dtype=np.float32)
        img = Normalized(img, self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:  # len(x.shape) mean the channel of that x
            mask = mask[:, :, 0]
        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis,
                                                          :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.names)

class STAREDatasetTest(Data.Dataset):

    def __init__(self, base_dir=r'../datasets/STARE',
                 mode='train', base_size=512, img_norm_cfg=None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('STARE', r'../datasets/STARE')
        else:
            self.img_norm_cfg = img_norm_cfg

        self.base_size = base_size
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('ppm'):
                self.names.append(filename)
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = Image.open(img_path).convert('I')
        mask = Image.open(label_path).convert('I')  # Convert to grayscale

        img = np.array(img, dtype=np.float32)  # Convert the data into numpy数组s
        img = Normalized(img, self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0  # Convert the data into numpy 数组 and normalization
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]  # Only take the first channel
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask

    def __len__(self):
        return len(self.names)

class CHASEDB1DatasetTrain(Data.Dataset):
    def __init__(self, base_dir=r'../datasets/CHASEDB1',
                 mode='train', base_size=960, patch_size = 256, img_norm_cfg = None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError


        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('CHASEDB1', '../datasets/CHASEDB1')
        else:
            self.img_norm_cfg = img_norm_cfg

        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('jpg'):
                self.names.append(osp.splitext(filename)[0])  # Store filename without extension
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.jpg') # Construct the mask path
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('L')  # Convert to grayscale
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.names)

class CHASEDB1DatasetTest(Data.Dataset):
    def __init__(self, base_dir=r'../datasets/CHASEDB1',
                 mode='train', base_size=960, img_norm_cfg = None):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg('CHASEDB1', '../datasets/CHASEDB1')
        else:
            self.img_norm_cfg = img_norm_cfg

        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('jpg'):
                self.names.append(osp.splitext(filename)[0])  # Store filename without extension
        self.tranform = augmentation()

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.jpg') # Construct the mask path
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('L')  # Convert to grayscale
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask

    def __len__(self):
        return len(self.names)


class NEUDatasetTrain(Data.Dataset):
    '''
    Return: Single channel
    '''

    def __init__(self, base_dir=r'../datasets/NEU',
                 mode='train', base_size=200, patch_size = 200):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('jpg'):
                self.names.append(osp.splitext(filename)[0])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.jpg')
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]

        img = torch.from_numpy(img_patch).type(torch.FloatTensor)
        mask = torch.from_numpy(mask_patch).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)

class NEUDatasetTest(Data.Dataset):
    def __init__(self, base_dir=r'../datasets/NEU',
                 mode='test', base_size=200, patch_size=200):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('jpg'):
                self.names.append(osp.splitext(filename)[0])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.jpg')
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask, [h, w]

    def __len__(self):
        return len(self.names)

class SaliencyDatasetTrain(Data.Dataset):
    def __init__(self, base_dir=r'../datasets/Saliency',
                 mode='train', base_size=200, patch_size=200):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('bmp'):
                self.names.append(osp.splitext(filename)[0])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.bmp')
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]

        img = torch.from_numpy(img_patch).type(torch.FloatTensor)
        mask = torch.from_numpy(mask_patch).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.names)

class SaliencyDatasetTest(Data.Dataset):

    def __init__(self, base_dir=r'../datasets/Saliency900',
                 mode='test', base_size=200, patch_size =200):
        assert mode in ['train', 'test']


        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'train')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.patch_size = patch_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('bmp'):
                self.names.append(osp.splitext(filename)[0])
    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name + '.bmp')
        label_path = osp.join(self.data_dir, 'masks', name + '.png')

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask, [h,w]

    def __len__(self):
        return len(self.names)




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

def PadImg(img, times=32):

    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        _, h, w = img.shape  # Unpack height, width, and channels
    else:
        raise ValueError("Unexpected number of dimensions in image")
    # h, w = img.shape
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