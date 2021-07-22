from collections import OrderedDict
import numpy as np
from . import util

def rms (a):
    return np.sqrt(np.mean(np.square(a)))

def nrms1D(im1,im2):
    #assumes numpy array input
    assert im1.shape == im2.shape, "images must have the same shape"  
    nrms = 200*rms(im1-im2)/(rms(im1)+rms(im2))
    return nrms

def batch_nrms(im1,im2):
    #aasumes tensor input
    
    assert im1.size() == im2.size(), "images must have the same shape"
    im1,im2=util.tensor2metric(im1),util.tensor2metric(im2)
    #### batch loop ###
    im_nrms=0
    channel_nrms=0
    batch_nrms=0
    for n in range(len(im1)):
        for c in range(len(im1[0])):
            for i in range(len(im1[0,0,0,:])):
                channel_nrms += nrms1D(im1[n,c,:,i],im2[n,c,:,i])
        
            im_nrms += channel_nrms/len(im1[0,0,0,:])
        batch_nrms+=im_nrms/len(im1[0])

    return batch_nrms/len(im1)

def batch_pearsonr(im1,im2):
    assert im1.shape == im2.shape, "images must have the same shape"
    im1,im2=util.tensor2metric(im1),util.tensor2metric(im2)
    #### batch loop ###
    pr=0
    channel_pr=0
    batch_pr=0
    for n in range(len(im1)):
        for c in range(len(im1[0])):
            for i in range(len(im1[0,0,0,:])):
                channel_pr += np.corrcoef(im1[n,c,:,i], im2[n,c,:,i])[0, 1]
            pr+=channel_pr/len(im1[0,0,0,:])
        batch_pr+=pr/len(im1[0])
    return batch_pr/len(im1)

#Function to pair train and validation of each metric
#inputs should be ordered dicts
def metric_wrapper(train_m,val_m):
    assert len(train_m)==len(val_m) , "both dictionaries must be the same size"
    pair_t_v=[] # is a list of ordered dicts
    for i, (k,v) in enumerate(train_m.items()):
        pair_t_v.append(OrderedDict())
        pair_t_v[i][k]=v
    for i, (k,v) in enumerate(val_m.items()):
        pair_t_v[i][k]=v
    return pair_t_v