import torch.nn.functional as F
import torch
import torch.nn as nn


# Reference Spade Norm: https://github.com/nvlabs/spade/
class User_norm(nn.Module):
    def __init__(self, label_nc, norm_nc, hidden, noise_in=True, norm_type='instance'):
        super(User_norm, self).__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded
        nhidden = hidden
        self.noise_in = noise_in
        if noise_in == True:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc * 2, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # self.mlp_gamma=nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, User_guide, noise):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        if self.noise_in == True:
            user_guide = F.interpolate(User_guide, size=x.size()[2:], mode='nearest')
            user_noise = F.interpolate(noise, size=x.size()[2:], mode='nearest')
            actv = self.mlp_shared(torch.cat([user_noise, user_guide], 1))
        else:
            actv = self.mlp_shared(User_guide)

        # b,c,h,w=user_guide.shape
        # noise=torch.rand(b,c,h,w).cuda()
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class UserResnetBlock_sketch(nn.Module):
    """
    This class is the sketch branch
    """

    def __init__(self, fin, fout):
        super(UserResnetBlock_sketch, self).__init__()
        # Attributes
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fout, kernel_size=3, padding=1)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        self.norm_0 = User_norm(3, fmiddle, 32)
        # self.norm_1 =User_norm(3,fout)

    def forward(self, x, user_sketch, noise):
        dx = self.conv_0(self.actvn(self.norm_0(x, user_sketch, noise)))
        out = x + dx
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class UserResnetBlock_color(nn.Module):
    """
    This class is the color propagation branch
    """

    def __init__(self, fin, fout):
        super(UserResnetBlock_color, self).__init__()
        # create conv layers

        self.gated_se = nn.Sequential(
            nn.InstanceNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, fout, kernel_size=3, padding=1),
            nn.InstanceNorm2d(fout),
            nn.ReLU(),
            nn.Conv2d(fout, fout, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.convcolor_f = nn.Conv2d(fout, fout, kernel_size=3, padding=1)
        self.normcolor_f = nn.InstanceNorm2d(fout)
        self.accolor_f = nn.LeakyReLU(0.2)

        self.convcolor = nn.Sequential(
            nn.Conv2d(fin, fout, kernel_size=3, padding=1),
        )

        self.color_to_f = nn.Conv2d(3, fout, kernel_size=1, stride=1, padding=0)

        self.convn = nn.Sequential(
            nn.InstanceNorm2d(fin),
            nn.LeakyReLU(),
            nn.Conv2d(fin, fout, kernel_size=3, padding=1)
        )
        self.norm = User_norm(fin, fout, int(fout / 2), noise_in=False)

    def forward(self, x, user_skrtch, user_color, noise, mask):
        b, c, h, w = user_color.shape
        user_color_f = user_color
        if c == 3:
            user_color_f = F.interpolate(user_color_f, size=x.size()[2:], mode='nearest')
            user_color_f = self.color_to_f(user_color_f)
        gated_se = self.gated_se(user_skrtch)
        first_x = self.convn(x)
        out_with_sketch = first_x * gated_se
        dx = x + out_with_sketch

        mask_color = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        user_color_f = self.convcolor_f(user_color_f * mask_color)
        user_color_f = (1 - gated_se) * user_color_f
        user_color_f = self.normcolor_f(user_color_f)
        user_color_f = self.accolor_f(user_color_f)
        out_with_color = self.convcolor(self.actvn(self.norm(dx, user_color_f, noise)))
        out = dx + out_with_color

        out = x + out
        return out, user_color_f, gated_se

    def actvn_gated(self, x):
        return F.sigmoid(x)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Edit_block(nn.Module):
    def __init__(self, layer_num, in_c, out_c):
        """

        Args:
            layer_num: The number of edit blocks
            in_c: input channel number
            out_c: output channel number
        """
        super(Edit_block, self).__init__()
        sk_blocks = []
        co_bolcks = []
        self.layer = layer_num
        for _ in range(layer_num):
            Res_sk = UserResnetBlock_sketch(1, 1)
            Res_co = UserResnetBlock_color(in_c, out_c)
            sk_blocks.append(Res_sk)
            co_bolcks.append(Res_co)
        self.sk_blocks = nn.Sequential(*sk_blocks)
        self.co_bolcks = nn.Sequential(*co_bolcks)
        self.sketch_out = []

    def forward(self, input, sketch, color, noise, mask):
        b, c, h, w = input.shape
        in_f_s = torch.mean(input, 1).contiguous().view(b, 1, h, w)
        in_f_c = input
        out_s = in_f_s
        out_c = in_f_c
        in_color = color
        for i in range(self.layer):
            out_s = self.sk_blocks[i](out_s, sketch, noise)
            out_c, in_color, gated_se = self.co_bolcks[i](out_c, out_s, in_color, noise, mask)
            self.sketch_out = []

        out = [out_s, out_c + input, self.sketch_out]
        return out


class Edit(nn.Module):
    def __init__(self):
        super(Edit, self).__init__()
        self.e_128 = Edit_block(6, 64, 64)
        self.e_64 = Edit_block(5, 128, 128)
        self.e_32 = Edit_block(4, 256, 256)
        self.e_16 = Edit_block(3, 512, 512)
        self.e_8 = Edit_block(2, 512, 512)
        self.e_4 = Edit_block(1, 512, 512)

        # This loss is useless
        self.down_mode_128 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.down_model_64 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.down_model_32 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.down_model_16 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.down_model_8 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.down_model_4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, input, user_sketch, user_color, noise, mask):
        out_128 = self.e_128(input[0], user_sketch, user_color, noise, mask)
        out_64 = self.e_64(input[1], user_sketch, user_color, noise, mask)
        out_32 = self.e_32(input[2], user_sketch, user_color, noise, mask)
        out_16 = self.e_16(input[3], user_sketch, user_color, noise, mask)
        out_8 = self.e_8(input[4], user_sketch, user_color, noise, mask)
        out_4 = self.e_4(input[5], user_sketch, user_color, noise, mask)

        # This loss is useless
        # loss_128_sk = self.down_mode_128(out_128[0])
        # loss_64_sk = self.down_model_64(out_64[0])
        # loss_32_sk = self.down_model_32(out_32[0])
        # loss_16_sk = self.down_model_16(out_16[0])
        # loss_8_sk = self.down_model_8(out_8[0])
        # loss_4_sk = self.down_model_4(out_4[0])

        out = [out_128[1], out_64[1], out_32[1], out_16[1], out_8[1], out_4[1], input[5]]
        # loss = [loss_128_sk, loss_64_sk, loss_32_sk, loss_16_sk, loss_8_sk, loss_4_sk]
        # visual = [out_128[2], out_64[2], out_32[2], out_16[2], out_8[2], out_4[2]]

        # out_final = [out, None, visual]

        return out
