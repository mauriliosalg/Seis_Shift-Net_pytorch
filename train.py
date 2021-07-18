import time
import sys
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.util import Logger
from collections import OrderedDict

if __name__ == "__main__":

    sys.stdout = Logger()
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    if opt.val:
        train_loader, val_loader=data_loader.load_data()
        train_size = len(train_loader)
        print('#training images = %d' % train_size)
        print('#validation images = %d' % len(val_loader))
        val_losses=[]
        val_nrms=[]
        val_pearsonr=[]
        
    else:
        train_loader = data_loader.load_data()
        train_size = len(train_loader)
        print('#training images = %d' % train_size)

    train_losses=[]
    train_nrms=[]
    train_pearsonr=[]
    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0

    train_start_time=time.time()
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        total_loss = 0
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data) # it not only sets the input data with mask, but also sets the latent mask.
            #print('setup time: ', time.time()-iter_start_time)

            # Additonal, should set it before 'optimize_parameters()'.
            if total_steps % opt.display_freq == 0:
                if opt.show_flow:
                    model.set_show_map_true()

            model.optimize_parameters()
            #print('optimize: ', time.time()-iter_start_time)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.show_flow:
                    model.set_flow_src()
                    model.set_show_map_false()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / train_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_networks('latest')

            #CH: Get L1 losses to compute mean over epoch
            total_loss += visualizer.get_loss_value(model.get_current_losses(),'G_L1')

            iter_data_time = time.time()
            #print('final: ', time.time()-iter_start_time)

        #CH:compute mean over epoch
        tloss=total_loss/epoch_iter
        train_losses.append(tloss)
        print('train Loss: %.3f ' % (tloss)) 

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_networks('latest')
            if not opt.only_lastest:
                model.save_networks(epoch)

        #CH: Validation computation
        if opt.val:
            total_loss=0
            for i, data in enumerate(val_loader):
                model.set_input(data)
                model.validate()
                #get individual val loss
                loss = model.get_val_loss()
                total_loss += visualizer.get_loss_value(loss,'Val_L1')
            #mean val loss
            vloss=total_loss/i
            val_losses.append(vloss)
            print('Validation Loss: %.3f ' % (vloss))
            
            t_v_loss=OrderedDict()
            t_v_loss['train']=tloss
            t_v_loss['validation']=vloss
            visualizer.plot_mean_losses(epoch, opt, t_v_loss)
            
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    t_v_loss=OrderedDict()
    t_v_loss['train']=train_losses
    if opt.val:
        t_v_loss['validation']=val_losses
    #visualizer.save_loss_plot(t_v_loss,opt)
    print('Total elapsed time: %d min' % ((time.time()-train_start_time)/60))
