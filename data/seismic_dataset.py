#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
from data.base_dataset import BaseDataset
from data.seismic_folder import make_dataset
from PIL import Image

class SeismicDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = opt.dataroot
        #CH: options for sesimic dataset
        self.nseisimg=opt.nseisimg
        self.nlines=opt.nlines
        self.linemute=opt.linemute
        self.A_imgs, self.A_samples,self.A_mean,self.A_max = make_dataset(self.dir_A,self.nseisimg,self.nlines,self.linemute,opt.phase,opt.smode)
        
        if self.opt.offline_loading_mask:
            self.mask_folder = self.opt.training_mask_folder if self.opt.isTrain else self.opt.testing_mask_folder
            self.mask_paths = sorted(make_dataset(self.mask_folder))

        if opt.smode=='sequential' or 'reconstruction':
            transform_list = [transforms.ToTensor(), transforms.Normalize((self.A_mean),(self.A_max))]
        else:
            transform_list = [transforms.ToTensor(), transforms.Normalize((self.A_mean),(self.A_max)), transforms.RandomHorizontalFlip(p=0.5)]
        
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        
        A_sample = [item[index] for item in self.A_samples]
        A_sample_str='_'.join(str(s) for s in A_sample)
        A = self.A_imgs[index]
        """
        w, h = A.size
        if w < h:
            ht_1 = self.opt.loadSize * h // w
            wd_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)
        else:
            wd_1 = self.opt.loadSize * w // h
            ht_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)   
        h = A.size(1)
        w = A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        """
        A = self.transform(A)
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = torch.flip(A, [2])

        # let B directly equals to A
        B = A.clone()
        A_flip = torch.flip(A, [2])
        B_flip = A_flip.clone()

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        if self.opt.offline_loading_mask:
            if self.opt.isTrain:
                mask = Image.open(self.mask_paths[random.randint(0, len(self.mask_paths)-1)])
            else:
                mask = Image.open(self.mask_paths[index % len(self.mask_paths)])
            mask = mask.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
                
        return {'A': A, 'B': B, 'A_F': A_flip, 'B_F': B_flip, 'M': mask,
                'A_sample': A_sample_str}

    def __len__(self):
        return len(self.A_imgs)

    def name(self):
        return 'SeismicDataset'
