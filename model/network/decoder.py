import torch
import torch.nn as nn

class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):
    def __init__(self, output_nc=3, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True)
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, 3, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)

        Decoder_1_2 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                               innermost=True)
        Decoder_2_2 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_3_2 = UnetSkipConnectionDBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_4_2 = UnetSkipConnectionDBlock(ngf * 4, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_5_2 = UnetSkipConnectionDBlock(ngf * 2, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_6_2 = UnetSkipConnectionDBlock(ngf, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

        # Texture Branch
        self.Decoder_1_2 = Decoder_1_2
        self.Decoder_2_2 = Decoder_2_2
        self.Decoder_3_2 = Decoder_3_2
        self.Decoder_4_2 = Decoder_4_2
        self.Decoder_5_2 = Decoder_5_2
        self.Decoder_6_2 = Decoder_6_2

    def forward(self, input_sgb):
        out_2_1 = self.Decoder_1_2(input_sgb[6])
        out_2_2 = self.Decoder_2_2(out_2_1)
        out_2_3 = self.Decoder_3_2(out_2_2)
        out_2_4 = self.Decoder_4_2(out_2_3)
        out_2_5 = self.Decoder_5_2(out_2_4)

        y_1 = self.Decoder_1(input_sgb[5]) + out_2_1
        y_2 = self.Decoder_2(torch.cat([y_1, input_sgb[4]], 1)) + out_2_2
        y_3 = self.Decoder_3(torch.cat([y_2, input_sgb[3]], 1)) + out_2_3
        y_4 = self.Decoder_4(torch.cat([y_3, input_sgb[2]], 1)) + out_2_4
        y_5 = self.Decoder_5(torch.cat([y_4, input_sgb[1]], 1)) + out_2_5
        y_6 = self.Decoder_6(torch.cat([y_5, input_sgb[0]], 1))

        return y_6