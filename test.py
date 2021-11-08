import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from collections import OrderedDict
import numpy as np

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!
    if opt.dataset_mode == 'seismic':
        isseismic=True
    else:
        isseismic=False

    test_start_time=time.time()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    
    #metrics preparation
    test_metrics=OrderedDict()

    for i, data in enumerate(dataset):
        
        if i >= opt.how_many:
            break
        t1 = time.time()
        model.set_input(data)
        model.test()
        t2 = time.time()
        print(t2-t1)
        if opt.metrics:
                model.compute_metrics()
                current_metrics = model.get_current_metrics()
                if i==0:
                    for key in current_metrics.keys():
                        test_metrics[key]=[]
                for key in current_metrics.keys():
                    test_metrics[key].append(current_metrics[key])
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, isseismic=isseismic)
    webpage.save()
    print(web_dir)
    print("Number of Images Processed: %d" % (i+1))
    print("total elapsed time: %d min" % ((time.time()-test_start_time)/60))
    if opt.metrics:
        file = open(os.path.join(web_dir,'results.txt'), "w")
        for key in test_metrics.keys():
            print('The mean value of the '+key+'metric for the test set was: {}'.format(np.array(test_metrics[key]).mean()))
            file.write('The mean value of the '+key+'metric for the test set was: {} \n'.format(np.array(test_metrics[key]).mean()))
        file.write('The total number of images processed was: {} \n'.format(i+1))
        file.close()
    