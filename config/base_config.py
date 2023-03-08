import argparse
import os
from util import util
import torch

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BaseConfig:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument(
            "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
        )
        parser.add_argument(
            "--gt_root",
            type=str,
            default=" ",
            help="path to detail images (which are the groundtruth)",
        )
        parser.add_argument(
            "--mask_root",
            type=str,
            default=" ",
            help="path to mask, we use the datasetsets of partial conv hear",
        )
        parser.add_argument(
            "--color_root",
            type=str,
            default=" ",
            help="path to color image, we use the datasetsets of partial conv hear",
        )
        parser.add_argument("--mask_type", type=str, default="random" , help="you can choose random or center, when you choose random, you need set the mask root",)
        parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
        parser.add_argument(
            "--num_workers", type=int, default=4, help="numbers of the core of CPU"
        )
        parser.add_argument(
            "--name",
            type=str,
            default="face",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--image_size",
            type=int,
            default=256,
            help="image size of training process",
        )
        parser.add_argument(
            "--input_nc", type=int, default=9, help="# of input image channels of encoder"
        )
        parser.add_argument(
            "--output_nc", type=int, default=3, help="# of output image channels"
        )
        parser.add_argument(
            "--input_nc_D",
            type=int,
            default=6,
            help="# of input image channels of discriminator",
        )
        parser.add_argument(
            "--ngf", type=int, default=64, help="# of gen filters in first conv layer"
        )
        parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in first conv layer",
        )
        parser.add_argument(
            "--n_layers_D",
            type=int,
            default=4,
            help="only used if which_model_netD==n_layers",
        )
        parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="DeFLOCNet",
            help="select the type of model",
        )
        parser.add_argument(
            "--nThreads", default=2, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="checkpoints",
            help="models and logs are saved here",
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="instance normalization or batch normalization",
        )
        parser.add_argument(
            "--use_dropout", action="store_true", help="use dropout for the generator"
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal|xavier|kaiming|orthogonal]",
        )

        parser.add_argument(
            "--lambda_L1", type=int, default=1, help="weight on L1 term in objective"
        )
        parser.add_argument(
            "--lambda_S", type=int, default=10, help="weight on Style loss in objective"
        )
        parser.add_argument(
            "--lambda_P",
            type=int,
            default=10,
            help="weight on Perceptual loss in objective",
        )
        parser.add_argument(
            "--lambda_Gan", type=int, default=1, help="weight on GAN term in objective"
        )
        parser.add_argument(
            "--lambda_TV", type=int, default=0.05, help="weight on TV loss in objective"
        )
        parser.add_argument(
            "--lambda_feat",
            type=float,
            default=10.0,
            help="weight for feature matching loss",
        )
        parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal, xavier and orthogonal.",
        )


        parser.add_argument("--num_D", default=2, type=int, help="D num")

        parser.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )


        # data_process
        parser.add_argument(
            "--need_crop", action="store_true", help="if true, cropping the images"
        )
        parser.add_argument(
            "--need_flip", action="store_true", help="if true, flipping the images"
        )
        self.initialized = True
        return parser

    def gather_config(self):
        # initialize parser with basic cfgions
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_config(self, cfg):
        message = ""
        message += "----------------- Config ---------------\n"
        for k, v in sorted(vars(cfg).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(basic_dir, cfg.checkpoints_dir, cfg.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "cfg.txt")
        with open(file_name, "wt") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

    def create_config(self):

        cfg = self.gather_config()
        cfg.isTrain = self.isTrain  # train or test

        # process cfg.suffix

        self.print_config(cfg)

        # set gpu ids
        str_ids = cfg.gpu_ids.split(",")
        cfg.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                cfg.gpu_ids.append(id)
        if len(cfg.gpu_ids) > 0:
            torch.cuda.set_device(cfg.gpu_ids[0])

        self.cfg = cfg
        return self.cfg
        # self.dataroot = "./datasets/celaba"  # 名称
        # self.batchSize = 1  # 尺寸
        # self.loadSize = 350
        # self.image_size = 256
        # self.input_nc = 3
        # self.output_nc = 3
        # self.ngf = 64
        # self.ndf = 64
        # self.which_model_netD = 'basic'
        # self.which_model_netG = 'unet_shift_triple'
        # self.triple_weight = 1
        # self.name = 'exp_unet_shift_triple'
        # self.n_layers_D = '3'
        # self.gpu_ids = [0]
        # self.dataset_mode = 'aligned'
        # self.model = 'CSA2_inpainting'
        # self.nThreads = 2
        # self.checkpoints_dir = './checkpoints'
        # self.norm = 'instance'
        # self.serial_batches = 'store_true'
        # self.display_winsize = 256
        # self.display_id = 1
        # self.display_port = 8097
        # self.suffix = ''
        # self.use_dropout = False
        # self.max_dataset_size = float("inf")
        # self.resize_or_crop = 'resize_and_crop'
        # self.no_flip = False
        # self.init_type = 'normal'
        # self.mask_type = 'random'
        # self.fixed_mask = 1
        # self.lambda_A = 100
        # self.threshold = 5 / 16.0
        # self.stride = 1
        # self.shift_sz = 1
        # self.mask_thred = 1
        # self.bottleneck = 512
        # self.gp_lambda = 10.0
        # self.ncritic = 5
        # self.constrain = 'MSE'
        # self.strength = 1
        # self.init_gain = 0.02
        # self.skip = 0
        # self.gan_type = 'lsgan'
        # self.gan_weight = 0.2
        # self.overlap = 4
        # self.display_freq = 10
        # self.print_freq = 50
        # self.display_single_pane_ncols = 0
        # self.save_latest_freq = 5000
        # self.save_epoch_freq = 1
        # self.continue_train = False
        # self.epoch_count = 1
        # self.phase = 'train'
        # self.which_epoch = ''
        # self.niter = 20
        # self.niter_decay = 100
        # self.beta1 = 0.5
        # self.lr = 0.0002
        # self.pool_size = 50
        # self.no_html = False
        # self.lr_policy = 'lambda'
        # self.lr_decay_iters = 50
        # self.update_html_freq = 1000
        # self.isTrain = True