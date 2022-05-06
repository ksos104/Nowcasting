import torch
import torch.nn as nn


def encoder_block(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1):
    encoder_blk = nn.Sequential(
        nn.Conv2d(in_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hidden_channel),
        nn.ReLU(),
        nn.Conv2d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(hidden_channel),
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


class Tunet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=16):
        super().__init__()
        self.encoder0 = encoder_block(in_channel, hidden_channel)
        self.skip0 = nn.Conv2d(hidden_channel,hidden_channel,kernel_size=1)
        self.maxPool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = encoder_block(hidden_channel, hidden_channel*2)
        self.skip1 = nn.Conv2d(hidden_channel*2,hidden_channel*2,kernel_size=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = encoder_block(hidden_channel*2, hidden_channel*4)
        self.skip2 = nn.Conv2d(hidden_channel*4,hidden_channel*4,kernel_size=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = encoder_block(hidden_channel*4, hidden_channel*8)
        self.skip3 = nn.Conv2d(hidden_channel*8,hidden_channel*8,kernel_size=1)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = encoder_block(hidden_channel*8, hidden_channel*32)

        self.convT0 = nn.ConvTranspose2d(hidden_channel*32, hidden_channel*8, kernel_size=2, stride=2)
        self.decoder0 = decoder_block(hidden_channel*16, hidden_channel*8)
        self.convT1 = nn.ConvTranspose2d(hidden_channel*8, hidden_channel*4, kernel_size=2, stride=2)
        self.decoder1 = decoder_block(hidden_channel*8, hidden_channel*4)
        self.convT2 = nn.ConvTranspose2d(hidden_channel*4, hidden_channel*2, kernel_size=2, stride=2)
        self.decoder2 = decoder_block(hidden_channel*4, hidden_channel*2)
        self.convT3 = nn.ConvTranspose2d(hidden_channel*2, hidden_channel, kernel_size=2, stride=2)
        self.decoder3 = decoder_block(hidden_channel*2, hidden_channel)

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
        concat0 = torch.cat([self.skip3(encoded3), transposed0], dim=1)
        decoded0 = self.decoder0(concat0)
        transposed1 = self.convT1(decoded0)
        concat1 = torch.cat([self.skip2(encoded2), transposed1], dim=1)
        decoded1 = self.decoder1(concat1)
        transposed2 = self.convT2(decoded1)
        concat2 = torch.cat([self.skip1(encoded1), transposed2], dim=1)
        decoded2 = self.decoder2(concat2)
        transposed3 = self.convT3(decoded2)
        concat3 = torch.cat([self.skip0(encoded0), transposed3], dim=1)
        decoded3 = self.decoder3(concat3)

        output = self.output_layer(decoded3)

        return output       ## output.shape = [bs, M, H, W]; bs = batch size (=1), M = num of output frames (=12)


if __name__ == '__main__':
    model = Tunet(13, 12)    ## unet(num_input_frames, num_output_frames)
    print(model)

    input = torch.rand((1, 13, 384, 384))
    output = model(input)
    print(output.shape)