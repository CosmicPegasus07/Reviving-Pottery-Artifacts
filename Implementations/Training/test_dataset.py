import os
import cv2
import numpy as np
import torch
import imgcrop
import random
import math
from PIL import Image, ImageDraw

from torchvision import transforms
from torch.utils.data import Dataset

import utils

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_valid_files(directory):
    files = utils.get_files(directory)
    return [f for f in files if is_image_file(f) and not os.path.basename(f).lower() == 'desktop.ini']

class InpaintDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # Filter out system files and non-image files from both directories
        image_files = get_valid_files(opt.baseroot)
        mask_files = get_valid_files(opt.baseroot_mask)
        
        # Sort numerically
        self.imglist = sorted(image_files, key=lambda d:int(d.split('/')[-1].split('.')[0]))
        self.masklist = sorted(mask_files, key=lambda d:int(d.split('/')[-1].split('.')[0]))

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        mask = cv2.imread(self.masklist[index])[:, :, 0]
        # find the Minimum bounding rectangle in the mask
        '''
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return img, mask
