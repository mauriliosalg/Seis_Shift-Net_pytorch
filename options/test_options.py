from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--testing_mask_folder', type=str, default='masks/testing_masks', help='perpared masks for testing')
        #CH: reconstruction options
        parser.add_argument('--recon_voverlap',type=int,default=128, help='vertical ovelap between patches in reconstrution mode')
        parser.add_argument('--data_recon_dir', type=str, default='./', help='dir to store the reconstructed data')
        parser.add_argument('--data_recon_name', type=str, default='dado_cut_recon',help='name of the reconstructed seismic data')
        parser.add_argument('--plot_line', type=int , default=330, help='line to save an image of before and after reconstruction')
        parser.add_argument('--save_numpy', action='store_true', help='save numpy images for figures generation')
        parser.add_argument('--save_recon', action='store_true', help='save numpy images for figures generation')
        parser.add_argument('--sgy_recon_dir', type=str, default='/scratch/maurilio/sismica_recon', help='dir to store the reconstructed data sgy file')
        self.isTrain = False

        return parser
