import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

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

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.forward = self.forward_split_cat

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class pconv_block(nn.Module):
    def __init__(self, dim, drop_path=0.2):
        super(pconv_block, self).__init__()
        self.partialconv = Partial_conv3(dim)
        self.norm = nn.InstanceNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.partialconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = pconv_block(dim)
        self.conv2 = pconv_block(dim)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.downsample_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.downsample_layer(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upsample_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.upsample_layer(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes=1, dims=None):
        super(UNet, self).__init__()
        if dims is None:
            dims = [32, 64, 128, 256]

        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=1)
        self.encoder1 = DoubleConv(dims[0])
        self.down1 = DownConv(dims[0], dims[1])
        self.encoder2 = nn.Sequential(DoubleConv(dims[1]), DoubleConv(dims[1]))
        self.down2 = DownConv(dims[1], dims[2])
        self.encoder3 = nn.Sequential(DoubleConv(dims[2]), DoubleConv(dims[2]))
        self.down3 = DownConv(dims[2], dims[3])

        self.bottleneck = nn.Sequential(DoubleConv(dims[3]), DoubleConv(dims[3]), DoubleConv(dims[3]),
                                        DoubleConv(dims[3]))
        self.up1 = UpConv(dims[3], dims[2])
        self.decoder1 = DoubleConv(dims[2])
        self.up2 = UpConv(dims[2], dims[1])
        self.decoder2 = DoubleConv(dims[1])
        self.up3 = UpConv(dims[1], dims[0])
        self.decoder3 = DoubleConv(dims[0])
        self.br = LBROM(dims[0], n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder1(x)
        out1 = x
        x = self.down1(x)
        x = self.encoder2(x)
        out2 = x
        x = self.down2(x)
        x = self.encoder3(x)
        out3 = x
        x = self.down3(x)

        x = self.bottleneck(x)
        x = self.up1(x)
        x_dec = x + out3
        x = self.decoder1(x_dec)
        x = self.up2(x)
        x_dec = x + out2
        x = self.decoder2(x_dec)
        x = self.up3(x)
        x_dec = x + out1
        x = self.decoder3(x_dec)
        x = self.br(x)
        return x


if __name__ == '__main__':
    from thop import profile
    net = UNet(1)
    x = torch.zeros(1, 1, 64, 48, 48)
    y = net(x)
    flops, params = profile(net, (x,))
    print(flops / 1e9)
    print(params / 1e6)