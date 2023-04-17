import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax

from model.SEAttention import SEAttention


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(

            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.pam=PAM(512)
        self.down4 = Down(512, 512)
        self.fpm=FPM()

        self.up1 = Up(1024, 256, bilinear)
        self.cb1 = SEAttention(channel=256,reduction=8)
        self.up2 = Up(512, 128, bilinear)
        self.cb2 = SEAttention(channel=128,reduction=8)
        self.up3 = Up(256, 64, bilinear)
        self.cb3 = SEAttention(channel=64,reduction=8)
        self.up4 = Up(128, 64, bilinear)
        self.cb4 = SEAttention(channel=64,reduction=8)
        self.outc = OutConv(64, n_classes)

    def forward(self, x1,x2):
        x=torch.cat([x1,x2],1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4=self.pam(x4)
        x5 = self.down4(x4)
        # x5 = self.pam1(x5)
        x5=self.fpm(x5)
        x = self.up1(x5, x4)
        x=self.cb1(x)
        x = self.up2(x, x3)
        x = self.cb2(x)
        x = self.up3(x, x2)
        x = self.cb3(x)
        x = self.up4(x, x1)
        x = self.cb4(x)
        logits = self.outc(x)
        return torch.sigmoid(logits)
class PAM(Module):
    """
    This code refers to "Dual attention network for scene segmentation"Position attention module".
    Ref from SAGAN
    """
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class FPM(nn.Module):
        def __init__(self, channels=512):
            """
            Feature Pyramid Attention
            :type channels: int
            """
            super(FPM, self).__init__()
            channels_mid = int(channels / 4)

            self.channels_cond = channels

            self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
            self.bn_master = nn.BatchNorm2d(channels)

            # Feature Pyramid
            self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3,
                                       bias=False)
            self.bn1_1 = nn.BatchNorm2d(channels_mid)
            self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
            self.bn2_1 = nn.BatchNorm2d(channels_mid)
            self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
            self.bn3_1 = nn.BatchNorm2d(channels_mid)

            self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
            self.bn1_2 = nn.BatchNorm2d(channels_mid)
            self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
            self.bn2_2 = nn.BatchNorm2d(channels_mid)
            self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
            self.bn3_2 = nn.BatchNorm2d(channels_mid)

            # Upsample
            self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                      bias=False)
            self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1,
                                                      bias=False)
            self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1,
                                                      bias=False)
            self.bn_upsample_1 = nn.BatchNorm2d(channels)

            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            """
            :param x: Shape: [b, 2048, h, w]
            :return: out: Feature maps. Shape: [b, 2048, h, w]
            """

            x_master = self.conv_master(x)
            x_master = self.bn_master(x_master)

            # Branch 1
            x1_1 = self.conv7x7_1(x)
            x1_1 = self.bn1_1(x1_1)
            x1_1 = self.relu(x1_1)
            x1_2 = self.conv7x7_2(x1_1)
            x1_2 = self.bn1_2(x1_2)

            # Branch 2
            x2_1 = self.conv5x5_1(x1_1)
            x2_1 = self.bn2_1(x2_1)
            x2_1 = self.relu(x2_1)
            x2_2 = self.conv5x5_2(x2_1)
            x2_2 = self.bn2_2(x2_2)

            # Branch 3
            x3_1 = self.conv3x3_1(x2_1)
            x3_1 = self.bn3_1(x3_1)
            x3_1 = self.relu(x3_1)
            x3_2 = self.conv3x3_2(x3_1)
            x3_2 = self.bn3_2(x3_2)

            # Merge branch 1 and 2
            x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))

            diffY = torch.tensor([ x3_upsample.size()[2] - x2_2.size()[2]])
            diffX = torch.tensor([ x3_upsample.size()[3] -x2_2.size()[3]])

            x2_2 = F.pad(x2_2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            x2_merge = self.relu(x2_2 + x3_upsample)

            x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))

            diffY = torch.tensor([x2_upsample.size()[2] - x1_2.size()[2]])
            diffX = torch.tensor([x2_upsample.size()[3] - x1_2.size()[3]])

            x1_2 = F.pad(x1_2, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

            x1_merge = self.relu(x1_2 + x2_upsample)

            x=self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

            diffY = torch.tensor([x_master.size()[2] - x.size()[2]])
            diffX = torch.tensor([x_master.size()[3] - x.size()[3]])

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

            x_master = x_master*x

            out = self.relu(x_master)

            return out
if __name__ == '__main__':
    net = UNet(n_channels=6, n_classes=1)
    im1=torch.randn(3,3,404,288)
    im2=torch.randn(3,3,404,288)
    out=net(im1,im2)
    print(out.shape)