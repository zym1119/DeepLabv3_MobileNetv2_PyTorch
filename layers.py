import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, s=1):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        """
        super(InvertedResidual, self).__init__()

        self.in_ = in_channels
        self.out_ = out_channels
        self.t = t
        self.s = s
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
                               stride=self.s, padding=1, groups=self.in_*self.t, bias=False))
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
        # if self.s == 1:
        #     # use residual connection
        #     if self.res_conv is None:
        #         out = x + self.block(x)
        #     else:
        #         out = self.res_conv(x) + self.block(x)
        # else:
        #     # plain block
        #     out = self.block(x)
        out = self.block(x)

        return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
    block = []
    block.append(InvertedResidual(in_, out_, t, s=s))
    for i in range(n-1):
        block.append(InvertedResidual(out_, out_, t, 1))
    return block
