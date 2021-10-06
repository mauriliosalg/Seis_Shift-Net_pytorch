import matplotlib.pyplot as plt
import numpy as np
import os

dirpath='/scratch/maurilio/log/teste/'
filename='50_train_loss.npy'
a=np.load(os.path.join(dirpath,filename))
epochs=np.arange(start=1,stop=len(a)+1,step=1)
figsize=(20,20)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k', squeeze=False, sharex=True)
axs[0,0].plot(epochs,a)
fig.savefig('teste.eps',format='eps')
fig.savefig('teste.png',format='png')
