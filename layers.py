import torch.nn as nn
import torch


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, s=1, dilation=1):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        :param dilation: dilation rate of 3*3 depthwise conv
        """
        super(InvertedResidual, self).__init__()

        self.in_ = in_channels
        self.out_ = out_channels
        self.t = t
        self.s = s
        self.dilation = dilation
        self.inverted_residual_block()

    def inverted_residual_block(self):
        """
        Build Inverted Residual Block and residual connection
        """
        block = []
        # pad = 1 if self.s == 3 else 0
        # conv 1*1
        block.append(nn.Conv2d(self.in_, self.in_*self.t, 1, bias=False))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv 3*3 depthwise
        block.append(nn.Conv2d(self.in_*self.t, self.in_*self.t, 3,
                               stride=self.s, padding=self.dilation, groups=self.in_*self.t, dilation=self.dilation,
                               bias=False))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv 1*1 linear
        block.append(nn.Conv2d(self.in_*self.t, self.out_, 1, bias=False))
        block.append(nn.BatchNorm2d(self.out_))

        self.block = nn.Sequential(*block)

        # if use conv residual connection
        if self.in_ != self.out_ and self.s != 2:
            self.res_conv = nn.Sequential(nn.Conv2d(self.in_, self.out_, 1, bias=False),
                                          nn.BatchNorm2d(self.out_))
        else:
            self.res_conv = None

    def forward(self, x):
        if self.s == 1:
            # use residual connection
            if self.res_conv is None:
                out = x + self.block(x)
            else:
                out = self.res_conv(x) + self.block(x)
        else:
            # plain block
            out = self.block(x)

        return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
    block = []
    block.append(InvertedResidual(in_, out_, t, s=s))
    for i in range(n-1):
        block.append(InvertedResidual(out_, out_, t, 1))
    return block


class ASPP_plus(nn.Module):
    def __init__(self, params):
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 1, bias=False),
                                     nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[0], dilation=params.aspp[0], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[1], dilation=params.aspp[1], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[2], dilation=params.aspp[2], bias=False),
                                      nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                      nn.BatchNorm2d(256))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)
    def forward(self, x):
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)

        # concate
        concate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)

        return self.concate_conv(concate)