import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch
import pywt
import pywt.data
import numpy as np
class SoftDiceLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        numerator = torch.sum(y_pred*y_gt)
        denominator = torch.sum(y_pred*y_pred + y_gt*y_gt)
        return numerator/denominator


class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)
        self.DWT = DWT()

    def forward(self, x):
        # print("First2D")
        # print(x.shape)
        # out = self.DWT(x)
        # print(out.shape)
        out = self.first(x)
        # print(out.shape)
        
        
        # out = self.DWT(out)
        # print(out.shape)
        return out


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)
        self.DWT = DWT()

    def forward(self, x):
        # print("Encoder2D")
        # print(x.shape)
        out = self.DWT(x)
        # print(out.shape)
        out = self.encoder(out)
        # print(out.shape)
        # out = self.DWT(out)
        # print(out.shape)

        return out


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)
        self.DWT = DWT()
        self.IWT = IWT()

    def forward(self, x):
        # print("in")
        # print("center2D")
        out = self.DWT(x)

        out = self.center(out)
        # print(out.shape)
        out = self.IWT(out)
        # print(out.shape)
        return out


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)
        self.IWT = IWT()
    def forward(self, x):
        # print("decoder2D")
        
        # print(x.shape)
        
        out = self.decoder(x)
        # print(out.shape)
        out = self.IWT(out)
        # print(out.shape)
        return out


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        # print("Last2D")
        # print(x.shape)
        out = self.first(x)
        # print(out.shape)
        return out


class First3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder3D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=downsample_kernel),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)

def dwt_init(x):

    # x01 = x[:, :, 0::2, :] / 2
    # x02 = x[:, :, 1::2, :] / 2
    # x1 = x01[:, :, :, 0::2]
    # x2 = x02[:, :, :, 0::2]
    # x3 = x01[:, :, :, 1::2]
    # x4 = x02[:, :, :, 1::2]
    # x_LL = x1 + x2 + x3 + x4
    # x_HL = -x1 - x2 + x3 + x4
    # x_LH = -x1 + x2 - x3 + x4
    # x_HH = x1 - x2 - x3 + x4
    coeffs2 = pywt.dwt2(x.detach().cpu().numpy(), 'db2')
    LL, (LH, HL, HH) = coeffs2
    LL=torch.tensor(LL).cuda()
    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    return LL

def iwt_init(x):
    # r = 2
    # in_batch, in_channel, in_height, in_width = x.size()
    # #print([in_batch, in_channel, in_height, in_width])
    # out_batch, out_channel, out_height, out_width = in_batch, int(
    #     in_channel / (r ** 2)), r * in_height, r * in_width
    # x1 = x[:, 0:out_channel, :, :] / 2
    # x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    # x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    # x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    # # h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    # # h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    # # h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    # # h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    # h[:, :, 0::2, 0::2] = x1 
    # h[:, :, 1::2, 0::2] = x1 
    # h[:, :, 0::2, 1::2] = x1 
    # h[:, :, 1::2, 1::2] = x1 

    coef = x.detach().cpu().numpy(),(None,None,None)
    h = pywt.idwt2(coef,'db2')
    h=torch.tensor(h).cuda()
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
