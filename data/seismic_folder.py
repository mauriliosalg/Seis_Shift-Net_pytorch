###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
import segyio
from PIL import Image
import os
import os.path
import numpy as np

SEIS_EXTENSIONS = [
    '.segy', '.SEGY','sgy', "SGY"
]

NPY_EXTENSIONS = [
    '.npy', '.NPY','npy', "NPY"
]

def random_int_with_mute(total,mute,size=1000):
    #total: should be the total range
    #mute: should be the interval to mute
    lista=np.arange(total)
    lista_mutada=np.concatenate((lista[:mute[0]],lista[mute[1]:]))
    return np.random.choice(lista_mutada,size)

def is_seismic_file(filename):
    return any(filename.endswith(extension) for extension in SEIS_EXTENSIONS)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)

def make_imgs(lines,nimg,mode,size=256, mute=[264,401]):
    
    bcenter=720 #centro do buraco
    blines=mute[1]-mute[0]+1 #número de linhas com buraco
    vtimes=5 #número de vezes que um patch será colhido verticalmente
    l=len(lines)
    xl=len(lines[0])-size
    d=len(lines.T)-size
    rl=nimg//l
    imgs=[]
    
    if mode=='random': 
        #samples=[np.random.randint(0,high=l,size=nimg),np.random.randint(0,high=xl,size=nimg),np.random.randint(0,high=d,size=nimg)]
        samples=[random_int_with_mute(l, mute,size=nimg),np.random.randint(0,high=xl,size=nimg),np.random.randint(0,high=d,size=nimg)]
    
        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])
   
    if mode=='sequential':

        samples = [np.tile(np.arange(0,l,step=1,dtype='int'),rl+1)[0:nimg],
                   np.repeat(np.arange(0,xl,step=xl//(rl+1),dtype='int'),l)[0:nimg],
                   np.repeat(np.arange(0,d,step=d//(rl+1),dtype='int'),l)[0:nimg]]

        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])
    
    if mode == 'reconstruction':
        nimg=blines*vtimes*2 #(numero de linhas com buraco)*(numero de patches verticais)*(numero de buracos)
        samples = [np.repeat(np.arange(mute[0],mute[1]+1,step=1,dtype='int'),vtimes*2)[0:nimg],
                   np.tile(np.array([62,597]),blines*vtimes)[0:nimg],
                  np.tile(np.repeat(np.arange(0,640,step=128,dtype='int'),2),blines)[0:nimg]]
                   
        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])

    return imgs,samples

def make_imgs_synt(lines,nimg,mode,size=256, mute=[264,401]):
    
    #bcenter=720 #centro do buraco
    #blines=mute[1]-mute[0]+1 #número de linhas com buraco
    #vtimes=5 #número de vezes que um patch será colhido verticalmente
    l=len(lines)
    xl=len(lines[0])-size
    d=len(lines.T)-size
    rl=nimg//l
    imgs=[]
    
    if mode=='random': 
        samples=[np.random.randint(0,high=l,size=nimg),np.random.randint(0,high=xl,size=nimg),np.random.randint(0,high=d,size=nimg)]
        #samples=[random_int_with_mute(l, mute,size=nimg),np.random.randint(0,high=xl,size=nimg),np.random.randint(0,high=d,size=nimg)]
    
        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])
   
    if mode=='sequential':

        samples = [np.tile(np.arange(0,l,step=1,dtype='int'),rl+1)[0:nimg],
                   np.repeat(np.arange(0,xl,step=xl//(rl+1),dtype='int'),l)[0:nimg],
                   np.repeat(np.arange(0,d,step=d//(rl+1),dtype='int'),l)[0:nimg]]

        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])

    return imgs,samples
"""
    if mode == 'reconstruction':
        nimg=blines*vtimes*2 #(numero de linhas com buraco)*(numero de patches verticais)*(numero de buracos)
        samples = [np.repeat(np.arange(mute[0],mute[1]+1,step=1,dtype='int'),vtimes*2)[0:nimg],
                   np.tile(np.array([62,597]),blines*vtimes)[0:nimg],
                  np.tile(np.repeat(np.arange(0,640,step=128,dtype='int'),2),blines)[0:nimg]]
                   
        for i in range(nimg):
            imgs.append(lines[samples[0][i]][samples[1][i]:samples[1][i]+size].T[samples[2][i]:samples[2][i]+size])
"""

def make_train_imgs(lines,xlines,nimg,mode,size=256, mute=[264,401],mutex=[668,778]):

    imgs=[]
    samples=[]
    if mode=='random': 
        l=len(lines) 
        x=len(xlines)
        xl=len(lines[0])-size
        il=len(xlines[0])-size
        d=len(lines.T)-size
        
        line_samples=[random_int_with_mute(l, mute,size=int(nimg/2)),np.random.randint(0,high=xl,size=int(nimg/2)),np.random.randint(0,high=d,size=int(nimg/2))]

        xline_samples = [random_int_with_mute(x, mutex,size=int(nimg/2)),np.random.randint(0,high=il,size=int(nimg/2)),np.random.randint(0,high=d,size=int(nimg/2))]
        
        samples = [ np.concatenate((line_samples[0],xline_samples[0])),np.concatenate((line_samples[1],xline_samples[1])),np.concatenate((line_samples[2],xline_samples[2])) ]

        for i in range(int(nimg/2)):
            imgs.append(lines[line_samples[0][i]][line_samples[1][i]:line_samples[1][i]+size].T[line_samples[2][i]:line_samples[2][i]+size])
        for i in range(int(nimg/2)):
            imgs.append(xlines[xline_samples[0][i]][xline_samples[1][i]:xline_samples[1][i]+size].T[xline_samples[2][i]:xline_samples[2][i]+size])

    return imgs,samples

def make_train_imgs_synt(lines,xlines,nimg,mode,size=256):

    imgs=[]
    samples=[]
    if mode=='random':
        l=len(lines) 
        print(l)
        x=len(xlines)
        print(x)
        xl=len(lines[0])-size
        il=len(xlines[0])-size
        print(il)
        d=len(lines.T)-size

        line_samples=[np.random.randint(0,high=l,size=int(nimg/2)),np.random.randint(0,high=xl,size=int(nimg/2)),np.random.randint(0,high=d,size=int(nimg/2))]

        xline_samples=[np.random.randint(0,high=x,size=int(nimg/2)),np.random.randint(0,high=il,size=int(nimg/2)),np.random.randint(0,high=d,size=int(nimg/2))]
        
        samples = [ np.concatenate((line_samples[0],xline_samples[0])),np.concatenate((line_samples[1],xline_samples[1])),np.concatenate((line_samples[2],xline_samples[2])) ]

        for i in range(int(nimg/2)):
            imgs.append(lines[line_samples[0][i]][line_samples[1][i]:line_samples[1][i]+size].T[line_samples[2][i]:line_samples[2][i]+size])
        for i in range(int(nimg/2)):
            imgs.append(xlines[xline_samples[0][i]][xline_samples[1][i]:xline_samples[1][i]+size].T[xline_samples[2][i]:xline_samples[2][i]+size])

    return imgs,samples

def make_dataset(dir,nimg,nlines ,mute, phase, mode):
    ### must have only one seismic file in dir###
    seismic = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_seismic_file(fname) or is_numpy_file(fname):
                path = os.path.join(root, fname)
                seismic.append(path)

    print(seismic[0])

    if is_seismic_file(fname):
        with segyio.open(seismic[0], iline=193, xline=197) as f:
            f.mmap()
            seisdata = segyio.tools.collect(f.iline[:])
            tseis = seisdata[nlines+1:]
            seis = seisdata[:nlines+1]
            if phase == 'test': 
                if mode=='reconstruction':
                    imgs, samples = make_imgs(seis,nimg,mode=mode)
                else:
                    imgs , samples = make_imgs(tseis,nimg,mode=mode)
            else:
                xseis= segyio.tools.collect(f.xline[:])
                imgs , samples = make_train_imgs(seis,xseis,nimg,mode=mode)

    elif is_numpy_file(fname):
        seisdata= np.load(seismic[0])
        tseis = seisdata[nlines+1:]
        seis = seisdata[:nlines+1]
        if phase == 'test': 
            imgs , samples = make_imgs_synt(tseis,nimg,mode=mode)
        else:
            xseis= seisdata.swapaxes(0,1)
            imgs , samples = make_train_imgs_synt(seis,xseis,nimg,mode=mode)

    return imgs, samples, seis.mean(), seis.max(), seisdata
    
    


def default_loader(path):
    return Image.open(path).convert('RGB')


class SeismicFolder(data.Dataset):

    def __init__(self, root, transform=None, return_samples=False):
        imgs, samples = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 seismic in: " + root + "\n"
                               "Supported seismic extensions are: " +
                               ",".join(SEIS_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.samples = samples
        self.transform = transform
        self.return_samples = return_samples

    def __getitem__(self, index):
        img = self.imgs[index]
        sample = [item[index] for item in self.samples]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_samples:
            return img, sample
        else:
            return img
    def __len__(self):
        return len(self.imgs)
