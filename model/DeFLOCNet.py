# -*-coding:utf-8-*-
import torch
import random
from loguru import logger
from functools import reduce
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from .network import networks
from .loss import PerceptualLoss, StyleLoss, InnerLoss, GANLoss, TVloss


class DeFLOCNet(BaseModel):
    def __init__(self, opt):
        """

        Args:
            opt: the setting of our model
        """
        super().__init__(opt)

        self.opt = opt
        self.mask_global = torch.ByteTensor(self.opt.batchSize, 1,
                                            opt.image_size, opt.image_size).to(self.device)
        self.mask_type = opt.mask_type
        self.gMask_opts = {}
        self.fixed_mask = opt.fixed_mask if opt.mask_type == 'center' else 0
        if opt.mask_type == 'center':
            assert opt.fixed_mask == 1, "Center mask must be fixed mask!"

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.to(self.device)

        # load/define networks
        self.netEN, self.netDE, self.netSGB = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                                                opt.norm,
                                                                opt.use_dropout, opt.init_type, self.gpu_ids,
                                                                opt.init_gain)
        self.model_names = ['EN', 'DE', 'SGB']

        if self.isTrain:
            self.old_lr = opt.lr

            # Define Loss Function
            self.PerceptualLoss = PerceptualLoss()
            self.InnerLoss = InnerLoss()
            self.StyleLoss = StyleLoss()
            self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor, opt=self.opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            self.netGlobalD = networks.define_D(
                self.opt, opt.init_type, self.gpu_ids, opt.init_gain
            )
            self.netLocalD = networks.define_D(
                self.opt, opt.init_type, self.gpu_ids, opt.init_gain
            )
            self.model_names.append(['GlobalD', 'LocalD'])

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_EN = torch.optim.Adam(self.netEN.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DE = torch.optim.Adam(self.netDE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_SGB = torch.optim.Adam(self.netSGB.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GlobalD = torch.optim.Adam(self.netGlobalD.parameters(),
                                                      lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_LocalD = torch.optim.Adam(self.netLocalD.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_EN)
            self.optimizers.append(self.optimizer_DE)
            self.optimizers.append(self.optimizer_SGB)
            self.optimizers.append(self.optimizer_GlobalD)
            self.optimizers.append(self.optimizer_LocalD)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netEN)
            networks.print_network(self.netDE)
            networks.print_network(self.netSGB)
            if self.isTrain:
                networks.print_network(self.netGlobalD)
                networks.print_network(self.netLocalD)
            print('-----------------------------------------------')
        if not self.opt.demo:
            if not self.isTrain or opt.continue_train:
                print('Loading pre-trained network! And you choose the {} epoch pretrained model'.format(opt.which_epoch))
                self.load_networks(opt.which_epoch)
            print('-----------------------------------------------')
        else:
            print('You choose the demo!')

    def name(self):
        return 'DeFLOCNet'
        # DexiNed

    def set_input(self, input_img, sketch, color, mask, mask_color):
        """
        Args:
            input_img: input image
            sketch: the sketch image
            color: the color image
            mask: mask image
            mask_color: the mask of color image
        """
        self.input_noise = torch.rand(self.opt.batchSize, 3, self.opt.image_size, self.opt.image_size).to(self.device)
        self.input_img = input_img.to(self.device)
        self.gt = input_img.to(self.device)
        self.sketch = torch.add(torch.neg(sketch.float()), 1).float().to(self.device)  # 0,1 replace
        self.color = color.to(self.device)

        # Define the mask, we have two types of mask, the first is center and the second is the input mask.
        if self.opt.mask_type == 'center':
            self.mask_global.zero_()
            self.mask_global[:, :, int(self.opt.image_size / 4) + self.opt.overlap: int(self.opt.image_size / 2) + int(
                self.opt.image_size / 4) - self.opt.overlap, \
            int(self.opt.image_size / 4) + self.opt.overlap: int(self.opt.image_size / 2) + int(
                self.opt.image_size / 4) - self.opt.overlap] = 1
        elif self.opt.mask_type == 'random':
            self.mask_global = mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        self.mask = self.mask_global.expand(self.opt.batchSize, 3, self.mask_global.size(2),
                                            self.mask_global.size(3)).float().to(self.device)
        self.inv_mask = torch.add(torch.neg(self.mask.float()), 1).float().cuda()

        self.input_sketch = self.sketch * self.mask
        self.input_noise = self.input_noise * self.mask
        self.input_color = self.color * mask_color.to(self.device)
        self.input_img.narrow(1, 0, 1).masked_fill_(self.mask_global.bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_img.narrow(1, 1, 1).masked_fill_(self.mask_global.bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_img.narrow(1, 2, 1).masked_fill_(self.mask_global.bool(), 2 * 117.0 / 255.0 - 1.0)
        # define the local patch
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.gt_local = self.gt[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64].to(self.device)

    def forward(self):
        out_encoder = self.netEN(
            torch.cat([self.input_img, self.inv_mask, self.input_noise], 1))
        out_sgb = self.netSGB(out_encoder, self.input_sketch, self.input_color, self.input_noise,
                              self.mask_global.float())
        self.fake = self.netDE(out_sgb)
        self.final_out = self.fake * self.mask + self.gt * self.inv_mask

    # def test(self):
    #     self.fake_p_1, self.fake_p_2, self.fake_p_3, self.fake_p_4, self.fake_p_5, self.fake_p_6 = self.netEN(
    #         torch.cat([self.input_img, self.inv_mask, self.input_noise], 1))
    #     self.x_in = [self.fake_p_1, self.fake_p_2, self.fake_p_3, self.fake_p_4, self.fake_p_5, self.fake_p_6]
    #     self.E_out = self.netSGB(self.x_in, self.input_sketch, self.input_color, self.input_noise,
    #                              self.mask_global.float())
    #     self.x_out = self.E_out[0]
    #     self.x_loss = self.E_out[1]
    #     self.visual = self.E_out[2]
    #     self.fake_B = self.netDE(self.x_out[0], self.x_out[1], self.x_out[2], self.x_out[3], self.x_out[4],
    #                              self.x_out[5], self.fake_p_6)
    #     self.real_B = self.gt
    #
    #     self.real_sk = self.sketch
    #     self.fake_B = self.fake_B * self.mask + self.real_B * self.inv_mask

    def backward_D(self):
        real = self.gt
        fake = self.fake.detach()
        pred_fake_global, pred_real_global, pred_fake_local, pred_real_local = self.discriminate(self.mask.float(),
                                                                                                 fake, real)
        D_losses_fake = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_fake_local, False, for_discriminator=True),
                self.criterionGAN(pred_fake_global, False, for_discriminator=True)
            ],
        )
        D_losses_real = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_real_global, True, for_discriminator=True),
                self.criterionGAN(pred_real_local, True, for_discriminator=True)
            ],
        )
        self.lossD = D_losses_fake + D_losses_real
        self.lossD.backward(retain_graph=True)

    def discriminate(self, input_mask, fake_image, real_image):
        fake_concat = torch.cat([input_mask, fake_image], dim=1)
        real_concat = torch.cat([input_mask, real_image], dim=1)

        real_local = self.gt_local
        fake_local = fake_image[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        fake_and_real_local = torch.cat([fake_local, real_local], dim=0)

        discriminator_out_global = self.netGlobalD(fake_and_real)
        discriminator_out_local = self.netLocalD(fake_and_real_local)

        pred_fake_global, pred_real_global = self.divide_pred(discriminator_out_global)
        pred_fake_local, pred_real_local = self.divide_pred(discriminator_out_local)
        return pred_fake_global, pred_real_global, pred_fake_local, pred_real_local

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def backward_G(self):
        G_losses = {}
        real = self.gt
        fake = self.fake
        comp = self.final_out
        # First, for discriminator
        pred_fake_global, pred_real_global, pred_fake_local, pred_real_local = self.discriminate(self.mask.float(),
                                                                                                 fake, real)
        G_losses["GAN"] = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_fake_global, True, for_discriminator=False),
                self.criterionGAN(pred_fake_local, True, for_discriminator=False),
            ],
        )

        # Second, for generator
        # TV loss
        G_losses["TV"] = TVloss(comp, self.mask, "mean") * self.opt.lambda_TV
        G_losses["L1"] = self.criterionL1(self.inv_mask * fake,
                                          self.inv_mask * real) + 2 * self.criterionL1(
            self.mask * fake, self.mask * real) * self.opt.lambda_L1

        G_losses["Perceptual"] = self.PerceptualLoss(fake, real) + self.PerceptualLoss(comp, real) * self.opt.lambda_P
        # self.Inner_Loss = self.InnerLoss(self.x_loss, self.real_sk, self.real_B)
        G_losses["Style"] = (self.StyleLoss(fake, real) + self.StyleLoss(comp, real)) * self.opt.lambda_Style

        loss_G = sum(G_losses.values()).mean()
        self.lossG = G_losses
        # self.loss_G = self.loss_G_L1 + self.loss_G_GAN * 0.1 + self.Perceptual_loss * 0.05 + self.Style_Loss * 250 + self.TV_loss * 0.1

        loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_GlobalD.zero_grad()
        self.optimizer_LocalD.zero_grad()
        self.backward_D()
        self.optimizer_GlobalD.step()
        self.optimizer_LocalD.step()

        self.optimizer_EN.zero_grad()
        self.optimizer_DE.zero_grad()
        self.optimizer_SGB.zero_grad()
        self.backward_G()
        self.optimizer_EN.step()
        self.optimizer_DE.step()
        self.optimizer_SGB.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict(
            [
                ("L1", self.lossG["L1"]),
                ("TV", self.lossG["TV"]),
                ("Perceptual", self.lossG["Perceptual"]),
                ("Style", self.lossG["Style"]),
                ("Generator", self.lossG["GAN"]),
                ("Discriminator", self.lossD),
            ]
        )

    def get_current_visuals(self):

        return {
            "input_image": self.input_img,
            "mask": self.mask,
            "ground_truth": self.gt,
            "fake": self.fake,
            "comp": self.final_out,
            "sketch": self.input_sketch,
            "color": self.input_color
        }

    # def save(self, epoch):
    #     torch.save(self.netEN.state_dict(), r'./canshu/EN_epoch{}.pkl'.format(epoch))
    #     torch.save(self.netDE.state_dict(), r'./canshu/DE_epoch{}.pkl'.format(epoch))
    #     torch.save(self.netSGB.state_dict(), r'./canshu/netSGB_epoch{}.pkl'.format(epoch))
    #     torch.save(self.netD.state_dict(), r'./canshu/D_epoch{}.pkl'.format(epoch))
    #     torch.save(self.netF.state_dict(), r'./canshu/F_epoch{}.pkl'.format(epoch))
    #
    # def loadnature(self, epoch):
    #     self.netEN.load_state_dict(torch.load(r'./canshu/EN_epoch{}.pkl'.format(epoch)))
    #     self.netDE.load_state_dict(torch.load(r'./canshu/DE_epoch{}.pkl'.format(epoch)))
    #     self.netSGB.load_state_dict(torch.load(r'./canshu/netSGB_epoch{}.pkl'.format(epoch)))
    #     self.netEN.eval()
    #     self.netDE.eval()
    #     self.netSGB.eval()
    #
    # def loadface(self, epoch):
    #     self.netEN.load_state_dict(torch.load(r'./canshuface/EN_epoch{}.pkl'.format(epoch)))
    #     self.netDE.load_state_dict(torch.load(r'./canshuface/DE_epoch{}.pkl'.format(epoch)))
    #     self.netSGB.load_state_dict(torch.load(r'./canshuface/netSGB_epoch{}.pkl'.format(epoch)))
    #
    #     self.netEN.eval()
    #     self.netDE.eval()
    #     self.netSGB.eval()
