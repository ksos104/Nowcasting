import math
import warnings
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class paddtw(nn.Module):
    def __init__(self, padd_ch):
        super(paddtw, self).__init__()
        self.padd_ch = padd_ch

    def forward(self, x):
        B, C, H, W = x.shape
        padd = self.padd_ch - C
        zero = torch.zeros(B, padd, H, W)
        out = torch.concat((x, zero), dim=1)
        return out

class TimeWise(nn.Module):
    def __init__(self, in_ch, f_num, tw_dilation, device=None, dtype=None):
        super(TimeWise, self).__init__()
        self.tw_dilation = tw_dilation
        self.f_num = f_num
        self.twd_filter = Parameter(torch.empty((f_num * tw_dilation, in_ch//tw_dilation, 1, 1), device=device, dtype=dtype), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.twd_filter, a=math.sqrt(5))

    def _conv_forward(self, input: Tensor, weight: Tensor):
        return F.conv2d(input, weight, bias=None, stride=1)

    def forward(self, input: Tensor) -> Tensor:
        for i in range(0, self.tw_dilation):
            ft_out = self._conv_forward(input[:, i::self.tw_dilation, :, :], self.twd_filter[i * self.f_num:(i + 1) * self.f_num, :, :, :])
            if i == 0:
                out = ft_out
            else:
                out = torch.cat((ft_out, out), dim=1)

        return out


class TWDBlock(nn.Module):
    def __init__(self, in_ch, f_num, f_size, tw_dilation):
        super(TWDBlock, self).__init__()
        self.tw_dilation = tw_dilation
        self.f_num = f_num
        self.padd_ch = (((in_ch-1)//tw_dilation)+1) * tw_dilation
        self.padding = paddtw(self.padd_ch)
        self.depthwise = nn.Conv2d(self.padd_ch, self.padd_ch, kernel_size=f_size, stride=1, padding=f_size//2, dilation=1, groups=self.padd_ch, bias=False)
        self.timewise = TimeWise(self.padd_ch, f_num, tw_dilation)


    def forward(self, x):

        out = self.padding(x)
        out = self.depthwise(out)
        out = self.timewise(out)
        return out


# array = torch.ones(64, 25, 128, 128) #BCHW
# twd = TWDBlock(25, 7, 3, 1)
# print(twd(array).shape)



def encoder_block(in_channel, f_num, tw_dilation, kernel_size=3):
    encoder_blk = nn.Sequential(
        TWDBlock(in_channel, f_num, kernel_size, tw_dilation),
        nn.BatchNorm2d(f_num * tw_dilation),
        nn.ReLU(),
        TWDBlock(f_num * tw_dilation, f_num, kernel_size, tw_dilation),
        nn.BatchNorm2d(f_num * tw_dilation),
        nn.ReLU()
    )

    return encoder_blk


def decoder_block(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1):
    decoder_blk = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hidden_channel),
        nn.ReLU(),
        nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hidden_channel),
        nn.ReLU()
    )

    return decoder_blk


class twd_2_unet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=16):
        super().__init__()
        self.encoder0 = encoder_block(in_channel, f_num=8, tw_dilation=2)
        self.maxPool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = encoder_block(hidden_channel, f_num=16, tw_dilation=2)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = encoder_block(hidden_channel * 2, f_num=32, tw_dilation=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = encoder_block(hidden_channel * 4, f_num=64, tw_dilation=2)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = encoder_block(hidden_channel * 8, f_num=256, tw_dilation=2)

        self.convT0 = nn.ConvTranspose2d(hidden_channel * 32, hidden_channel * 8, kernel_size=2, stride=2)
        self.decoder0 = decoder_block(hidden_channel * 16, hidden_channel * 8)
        self.convT1 = nn.ConvTranspose2d(hidden_channel * 8, hidden_channel * 4, kernel_size=2, stride=2)
        self.decoder1 = decoder_block(hidden_channel * 8, hidden_channel * 4)
        self.convT2 = nn.ConvTranspose2d(hidden_channel * 4, hidden_channel * 2, kernel_size=2, stride=2)
        self.decoder2 = decoder_block(hidden_channel * 4, hidden_channel * 2)
        self.convT3 = nn.ConvTranspose2d(hidden_channel * 2, hidden_channel, kernel_size=2, stride=2)
        self.decoder3 = decoder_block(hidden_channel * 2, hidden_channel)

        self.output_layer = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

    def forward(self, x):
        ## x.shape = [bs, N, H, W]; bs = batch size (=1), N = num of input frames (=13)
        encoded0 = self.encoder0(x)
        encodedP0 = self.maxPool0(encoded0)
        encoded1 = self.encoder1(encodedP0)
        encodedP1 = self.maxPool1(encoded1)
        encoded2 = self.encoder2(encodedP1)
        encodedP2 = self.maxPool2(encoded2)
        encoded3 = self.encoder3(encodedP2)
        encodedP3 = self.maxPool3(encoded3)
        feature = self.center(encodedP3)

        transposed0 = self.convT0(feature)
        concat0 = torch.cat([encoded3, transposed0], dim=1)
        decoded0 = self.decoder0(concat0)
        transposed1 = self.convT1(decoded0)
        concat1 = torch.cat([encoded2, transposed1], dim=1)
        decoded1 = self.decoder1(concat1)
        transposed2 = self.convT2(decoded1)
        concat2 = torch.cat([encoded1, transposed2], dim=1)
        decoded2 = self.decoder2(concat2)
        transposed3 = self.convT3(decoded2)
        concat3 = torch.cat([encoded0, transposed3], dim=1)
        decoded3 = self.decoder3(concat3)

        output = self.output_layer(decoded3)

        return output  ## output.shape = [bs, M, H, W]; bs = batch size (=1), M = num of output frames (=12)




if __name__ == '__main__':
    model = twd_2_unet(13, 12)    ## unet(num_input_frames, num_output_frames)
    print(model)

    input = torch.rand((1, 13, 384, 384))
    output = model(input)
    print(output.shape)