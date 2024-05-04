import numpy as np
import os
import ntpath
import time
import sys
from subprocess import Popen, PIPE
from PIL import Image
from . import util, html
#from scipy.misc import imresize
from skimage.transform import resize
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def save_numpy(web_dir,visuals, image_path, media, maximo):
    image_dir=os.path.join(web_dir,'image_numpy')
    if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, im_data in visuals.items():
        im = util.tensor2metric(im_data)
        im=(im*maximo) + media       
        image_name = '%s_%s' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        np.save(save_path,im[0,0])

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256 , isseismic=False):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    
    image_dir = webpage.get_image_dir()
    
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        if isseismic:
            im = util.rm_extra_dim_seis(im_data)
            im = util.tensor2seis(im)
            
        else:
            im = util.tensor2im(im_data)
        
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        if isseismic:
            _, h, w = im.shape
            im=im[0]
        else:
            h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im =  resize(im, (h, int(w * aspect_ratio)), order=3)
            #im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im =  resize(im, (int(h / aspect_ratio), w), order=3)
            #im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

#CH: Plot and save seismic image
def plot_seis(image,clip=200,cmap=plt.cm.Greys,figname="secao_sismica"):
    figsize=(20,20)
    clip = 200
    vmin, vmax = -clip, clip
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k', squeeze=False, sharex=True)
    axs = axs.ravel()
    axs[0].imshow(image, cmap=plt.cm.Greys, vmin=vmin, vmax=vmax)
    plt.savefig(figname)

def plot_compara(im1,im2,clip=200,cmap=plt.cm.Greys,figname="secao_sismica"):
    figsize=(20,20)
    clip = 200
    vmin, vmax = -clip, clip
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='w', edgecolor='k', squeeze=False, sharex=True)
 
    im=axs[0][0].imshow(im1, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0][0].set_title('(a) Antes' , fontsize=20)
    axs[1][0].imshow(im2, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1][0].set_title('(b) Depois', fontsize=20)
    axs[1][0].set_xlabel('Crosslines', fontsize=20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #fig.text(0.5, 0.05, 'Crosslines', ha='center')
    fig.text(0.05, 0.5, 'Amostras em Profundidade', va='center', rotation='vertical', fontsize=20)
    
    plt.savefig(figname+'.eps',format='eps')
    plt.savefig(figname+'.png',format='png')

def save_seis_image(data,figname="secao_sismica.jpg"):
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(figname)

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.opt = opt
        self.dataset_mode=opt.dataset_mode
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                if self.dataset_mode != 'seismic':
                    for label, image in visuals.items():
                        image = util.rm_extra_dim(image) # remove the dummy dim
                        image_numpy = util.tensor2im(image)
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                        if idx % ncols == 0:
                            label_html += '<tr>%s</tr>' % label_html_row
                            label_html_row = ''
                    white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                    while idx % ncols != 0:
                        images.append(white_image)
                        label_html_row += '<td></td>'
                        idx += 1
                    if label_html_row != '':
                        label_html += '<tr>%s</tr>' % label_html_row
                    try:
                        self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                        padding=2, opts=dict(title=title + ' images'))
                        label_html = '<table>%s</table>' % label_html
                        self.vis.text(table_css + label_html, win=self.display_id + 2,
                                    opts=dict(title=title + ' labels'))
                    except VisdomExceptionBase:
                        self.create_visdom_connections()
                else:
                    for label, image in visuals.items():
                        image = util.rm_extra_dim_seis(image) # remove the dummy dim
                        image_numpy = util.tensor2seis(image)
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy)
                        idx += 1
                        if idx % ncols == 0:
                            label_html += '<tr>%s</tr>' % label_html_row
                            label_html_row = ''
                    white_image = np.ones_like(image_numpy)* 255
                    while idx % ncols != 0:
                        images.append(white_image)
                        label_html_row += '<td></td>'
                        idx += 1
                    if label_html_row != '':
                        label_html += '<tr>%s</tr>' % label_html_row
                    try:
                        self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                        padding=2, opts=dict(title=title + ' images'))
                        label_html = '<table>%s</table>' % label_html
                        self.vis.text(table_css + label_html, win=self.display_id + 2,
                                    opts=dict(title=title + ' labels'))
                    except VisdomExceptionBase:
                        self.create_visdom_connections()
                
            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        #import pdb
        #pdb.set_trace()
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
    
    def plot_mean_losses(self,epoch, opt, losses):
        if not hasattr(self, 'plot_data2'):
            self.plot_data2 = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data2['X'].append(epoch)
        self.plot_data2['Y'].append([losses[k] for k in self.plot_data2['legend']])
        self.vis.line(
            X=np.array(self.plot_data2['X']),
            Y=np.array(self.plot_data2['Y']),
            opts={
                'title': self.name + ' mean loss over time',
                'legend': self.plot_data2['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=4)
    #CH: function to plot metris        
    def plot_mean_metrics(self,epoch,opt,list_metrics):
        if not hasattr(self, 'plot_metrics'):
            self.plot_metrics=[]
            for m in list_metrics:
                self.plot_metrics.append({'X': [], 'Y': [], 'legend': list(m.keys())})
        
        for i,metrics in enumerate(list_metrics):
            
            self.plot_metrics[i]['X'].append(epoch)
            self.plot_metrics[i]['Y'].append([metrics[k] for k in self.plot_metrics[i]['legend']])
            
            self.vis.line(
                X=np.array(self.plot_metrics[i]['X']),
                Y=np.array(self.plot_metrics[i]['Y']),
                opts={
                    'title': self.name + ' mean metrics over time',
                    'legend': self.plot_metrics[i]['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'Metric Values'},
                win=5+i)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    #CH: funtion to print metrics
    def print_metrics(self, metrics):
        message = 'Metrics Results for epoch -> '
        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    #CH: function to return value of specific loss
    def get_loss_value(self,losses,key):
        for k, v in losses.items():
           if k==key:
               return v
    #CH: Save loss_plot
    def save_loss_plot(self,losses,opt):
        import pdb
        pdb.set_trace()
        fig, ax = plt.subplots(1,1, figsize=(14,12))
        for k, v in losses.items():
            ax.plot(np.arange(len(v)),np.array(v),label=k)
        ax.set_ylabel('Losses Values',fontsize=16 )
        ax.set_xlabel('Epochs',fontsize=16)
        ax.legend()
        ax.set_title('Train & Val Losses x Epochs', fontsize=20)
        plt.savefig(os.path.join(opt.checkpoints_dir, opt.name,"losses.jpg"))
    
    
