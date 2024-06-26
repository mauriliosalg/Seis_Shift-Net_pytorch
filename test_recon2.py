import time
import os
from typing import OrderedDict
import segyio
import numpy as np
from shutil import copyfile
from numpy.lib.function_base import median
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images, plot_seis, save_seis_image, plot_compara
from util import html
from util import util
import matplotlib.pyplot as plt

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!
    experiments_names=['Metrics_cw_124_rdl','Metrics_BR_cw_96_rdl','Metrics_cw_68_rdl']

    if opt.dataset_mode == 'seismic':
        isseismic=True
    else:
        isseismic=False

    test_start_time=time.time()
    data_loader = CreateDataLoader(opt)
    #seismic data and statistics
    sismica= getattr(data_loader.dataset, 'seis_data')
    media= getattr(data_loader.dataset, 'A_mean')
    maximo = getattr(data_loader.dataset, 'A_max')
    dataset = data_loader.load_data()
    #models chosen for reconstruction
    models=OrderedDict()
    for name in experiments_names:
        opt.name=name
        models[name] = create_model(opt)
    opt.name = 'recon'
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # folder to store the reconstructed seismic
    recon_dir= os.path.join(opt.data_recon_dir, 'recon_data')
    if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
    #CH: saving original seismic session for comparison
    im1=np.copy(sismica[opt.plot_line][:].T[:600])
    plot_seis(sismica[opt.plot_line][:].T[:],clip=200,cmap=plt.cm.Greys,figname=os.path.join(recon_dir,'line'+str(opt.plot_line)+'_before.jpg'))
    # test
    for i, data in enumerate(dataset):
        #image sample information splited
        sampleCoord=data['A_sample'][0].split(sep="_")
        line=int(sampleCoord[0])
        xline=int(sampleCoord[1])
        depth=int(sampleCoord[2])
        #select model for correct size of mask
        if xline==597:
            model=models[experiments_names[0]] #masksize=124
        if xline==62:
            model=models[experiments_names[1]] #masksize=96

        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        #generated image for reconstruction
        fakeB = util.tensor2metric(getattr(model,"fake_B")) #model output to numpy array
        fakeB = (fakeB*maximo) + media #unnormalize the image
        sismica[line][xline:xline+256].T[depth:depth+256]=fakeB #substitute image on original seismic file
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, isseismic=isseismic)

    ### CH: Plotting an inline reconstructed
    im2=sismica[opt.plot_line][:].T[:600]
    np.save(os.path.join(recon_dir,'line'+str(opt.plot_line)+'recon'),im2)
    plot_seis(sismica[opt.plot_line][:].T[:],clip=200,cmap=plt.cm.Greys,figname=os.path.join(recon_dir,'line'+str(opt.plot_line)+'_reconstructed.jpg'))
    plot_compara(im1,im2,clip=200,cmap=plt.cm.Greys,figname=os.path.join(recon_dir,'line'+str(opt.plot_line)+'_compara'))

    if opt.save_recon:
        
        output_file = os.path.join(opt.sgy_recon_dir,'dado_cut_recon2.sgy')
        segyio.tools.from_array3D(output_file, sismica, iline=193, xline=197, dt=5000,delrt=0)
        

    webpage.save()
    print(web_dir)
    print("total elapsed time: %d min" % ((time.time()-test_start_time)/60))