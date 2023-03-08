# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .encoder import Encoder
from .decoder import Decoder
from .structure_generation_block import Edit
from .discriminator import MultiscaleDiscriminator


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf,  norm='instance', use_dropout=False,
             init_type='normal', gpu_ids=[], init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)
    netE = Encoder(input_nc=input_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    netD = Decoder(output_nc=output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    SGB = Edit_block()
    return init_net(netE, init_type, init_gain, gpu_ids), init_net(netD, init_type, init_gain, gpu_ids), init_net(
        SGB, init_type, init_gain, gpu_ids)


def define_D(opt, init_type="normal", gpu_ids=[], init_gain=0.02):
    netD = MultiscaleDiscriminator(opt)
    return init_net(netD, init_type, init_gain, gpu_ids)


class Edit_block(nn.Module):
    def __init__(self):
        super(Edit_block, self).__init__()
        self.edit_block = Edit()

    def forward(self, input, sketch, color, noise, mask):
        out = self.edit_block(input, sketch, color, noise, mask)
        return out


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
