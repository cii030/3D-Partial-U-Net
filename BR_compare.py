import torch
from torch import nn
from thop import profile


class LBROM(nn.Module):
    def __init__(self, out_c, n_class):
        super(LBROM, self).__init__()
        self.transposeconv1 = nn.ConvTranspose3d(out_c, n_class, 3, 1, 1)
        self.pwconv1 = nn.Conv3d(n_class,4 * n_class, kernel_size=1, groups=n_class)
        self.dwconv = nn.Conv3d(4 * n_class, 4 * n_class, kernel_size=7, padding=3, groups=4 * n_class)
        self.norm = nn.BatchNorm3d(4 * n_class)
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Conv3d(4 * n_class, n_class, kernel_size=1, groups=n_class)

    def forward(self, x):
        x_left = self.transposeconv1(x)
        x_right = self.pwconv1(x_left)
        x_right = self.dwconv(x_right)
        x_right = self.norm(x_right)
        x_right = self.act(x_right)
        x_right = self.pwconv2(x_right)
        x = x_left + x_right
        return x

class BR(nn.Module):
    def __init__(self, out_c, n_class):
        super(BR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_c),
            nn.ReLU(out_c)
        )
        # self.conv1 = pconv_block(out_c, drop_path=0.2)
        self.transposeconv1 = nn.ConvTranspose3d(out_c, n_class, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(n_class, n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(n_class)
        )
        self.conv3 = nn.Conv3d(n_class, n_class, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x_left = self.conv1(x)
        x_left = self.transposeconv1(x_left)
        x_right = self.conv2(x_left)
        x_right = self.conv3(x_right)
        x = x_left + x_right
        return x

if __name__ == '__main__':
    block1 = BR(32, 1)
    block2 = LBROM(32, 1)
    input = torch.rand(1, 32, 64, 64, 48)
    output = block1(input)
    flops, params = profile(block1, (input,))
    print(flops / 1e9)
    print(params / 1e6)
    print(input.size(), output.size())
    flops2, params2 = profile(block2, (input,))
    print(flops2 / 1e9)
    print(params2 / 1e6)
    print(flops/flops2)
    print(params/params2)