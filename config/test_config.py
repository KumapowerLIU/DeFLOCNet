from .base_config import BaseConfig


class TestConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)
        parser.add_argument(
            "--ntest", type=int, default=float("inf"), help="# of test examples."
        )
        parser.add_argument(
            "--test_image", default=r" ", type=str, help="the dir of test image"
        )
        parser.add_argument(
            "--mask_image", default=r" ", type=str,
            help="the dir of mask image, the color mask image is same with image mask during testing"
        )
        parser.add_argument(
            "--color_image", default=r" ", type=str, help="the dir of test color image"
        )
        parser.add_argument(
            "--sample_num", type=int, default=float("inf"), help="# of test examples."
        )
        parser.add_argument(
            "--results_dir", type=str, default="./results/", help="saves results here."
        )
        parser.add_argument(
            "--which_epoch",
            type=str,
            default="120",
            help="which epoch to load? set to latest to use latest cached model",
        )
        parser.add_argument(
            "--model_save_path",
            type=str,
            default="checkpoints",
            help="the save path of pre-trained parameters ",
        )
        parser.add_argument(
            "--test_image_size",
            type=int,
            default=256,
            help="the image size of test processing",
        )
        parser.add_argument('--demo', default = True,
                             help="if true, using the GUI demo")

        self.isTrain = False
        return parser
