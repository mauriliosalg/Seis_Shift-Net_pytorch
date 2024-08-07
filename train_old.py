import time
import sys
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.util import Logger

if __name__ == "__main__":

    sys.stdout = Logger()
    
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0

    train_start_time=time.time()
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
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
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            #print('visualize: ', time.time()-iter_start_time)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
            #print('final: ', time.time()-iter_start_time)
            
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_networks('latest')
            if not opt.only_lastest:
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    
    print('Total elapsed time: %d min' % ((time.time()-train_start_time)/60))
