import matplotlib.pyplot as plt
import numpy as np
import os, fnmatch
from collections import OrderedDict



def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

metric_names=['nrms','loss','pearsonr','ssim','psnr']
dirpath='/scratch/maurilio/log/Metrics_cw_124_rdl/'
files=find('350*.npy', dirpath)
Train=OrderedDict()
Val=OrderedDict()
for file in files:
    
    if fnmatch.fnmatch(file, '*Val*') or fnmatch.fnmatch(file, '*val*'):
        metric=[name for name in metric_names if(name in file)]
        Val[metric[0]]=np.load(file)
    else:
        metric=[name for name in metric_names if(name in file)]
        Train[metric[0]]=np.load(file)
"""
filename='50_train_loss.npy'
a=np.load(os.path.join(dirpath,filename))
epochs=np.arange(start=1,stop=len(a)+1,step=1)
figsize=(20,20)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k', squeeze=False, sharex=True)
axs[0,0].plot(epochs,a)
fig.savefig('teste.eps',format='eps')
fig.savefig('teste.png',format='png')
"""
for name in metric_names:
    epochs=np.arange(start=1,stop=len(Train['loss'])+1,step=1)
    figsize=(19.2,10.8)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k', squeeze=False, sharex=True)
    axs[0,0].plot(epochs,Train[name],label="Curva de Treinamento")
    axs[0,0].plot(epochs,Val[name],label="Curva de Validação")
    axs[0,0].set_title(name.upper()+' para máscara centralizada com 124 pixels', fontsize=22)
    axs[0,0].set_xlabel('Número de Épocas', fontsize=22)
    if name in ['nrms','ssim','psnr']:
        axs[0,0].set_ylabel(name.upper()+ ' médio', fontsize=22)
    elif name == 'pearsonr':
        axs[0,0].set_ylabel('Pearson R médio', fontsize=22)
    else:
        axs[0,0].set_ylabel(name.upper()+ ' média', fontsize=22)
    axs[0,0].tick_params(axis = 'both', which = 'major', labelsize = 16)
    axs[0,0].legend( loc='upper right', fontsize=22,bbox_to_anchor=(0.9, 0.8))
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig.savefig(name+'_cw_124_rdl.eps',format='eps',dpi=300)
    #fig.savefig(name+'.png',format='png',dpi=300)
