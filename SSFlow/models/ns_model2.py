import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import numbers
from einops import rearrange
class KernelConv2D(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                Downsample(channels=in_channels, filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class BasicBlock_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_E, self).__init__()
        self.mode = mode

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )
        if mode == 'down':
            self.reshape_conv = ResidualDownSample(in_channels, out_channels)

    def forward(self, x):
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        if self.mode is not None:
            out = self.reshape_conv(out)
        return out

class BasicBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        if self.mode is not None:
            x = self.reshape_conv(x)
        res = self.body1(x)
        out = res + x
        out = self.body2(out)
        return out


class BasicBlock_D_2Res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mode=None, bias=True):
        super(BasicBlock_D_2Res, self).__init__()
        self.mode = mode
        if mode == 'up':
            self.reshape_conv = ResidualUpSample(in_channels, out_channels)

        self.body1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=bias)
        )

    def forward(self, x):
        if self.mode is not None:
            x = self.reshape_conv(x)
        res1 = self.body1(x)
        out1 = res1 + x
        res2 = self.body2(out1)
        out2 = res2 + out1
        return out2


## Channel Attention (CA) Layer
class CurveCALayer(nn.Module):
    def __init__(self, channel, n_curve):
        super(CurveCALayer, self).__init__()
        self.n_curve = n_curve
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(channel, n_curve, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # clip the input features into range of [0,1]
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x-1)
        for i in range(self.n_curve):
            x = x + a[:,i:i+1]*x*(1-x)
        return x

class KernelConv2D(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


import numbers
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FSAS1(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS1, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden1 = nn.Conv2d(dim*2, dim * 6*2, kernel_size=1, bias=bias)

        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.to_hidden_dw1 = nn.Conv2d(dim * 6*2, dim * 6*2, kernel_size=3, stride=1, padding=1, groups=dim *2* 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.project_out1 = nn.Conv2d(dim* 2 * 2, dim* 2, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

        self.downconv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)

    def forward(self, x,x1):
        hidden = self.to_hidden(x)#16,748,40,40
        hidden1 = self.to_hidden1(x1)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)#16,256,40,40
        q1, k1, v1 = self.to_hidden_dw1(hidden1).chunk(3, dim=1)


        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        out = self.downconv1(
            F.interpolate(out, scale_factor=1/2, mode='bilinear', align_corners=False,
                          recompute_scale_factor=True))

        output = v1 * out
        output = self.project_out1(output)

        return output

class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)


        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8


    def forward(self, x):
        hidden = self.to_hidden(x)#16,748,40,40

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)#16,256,40,40

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output



class OrthoNet(nn.Module):
#make 1000
    def __init__(self, block, layers, num_classes=1000):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inplanes = 64
        super(OrthoNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,64, layers[0])
        self.layer2 = self._make_layer(block, 128,32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256,16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512,8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(10816, 16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.to(self._device)

    def _make_layer(self, block, planes,height, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes,height, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,height))

        return nn.Sequential(*layers)

    def forward(self, x2):#16,64,80,80
        x = self.conv1(x2)#16,64,40,40
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#16,64,20,20

        x = self.layer1(x)
        x = self.layer2(x)#16,128,10,10
        x = self.layer3(x)#16,256,5,5
        x = self.layer4(x)#16,512,3,3

        x = self.avgpool(x)#16,512,3,3
        x = x.view(x2.size(2), -1)#16,4608
        x = self.fc(x)

        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

import model.attention as Attention
import model.transforms as Transforms

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._process: nn.Module = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), device=self.device, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, device=self.device, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention.Attention()
        self.F_C_A = Transforms.GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0), out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        attention = excitation * out
        attention += residual
        activated = torch.relu(attention)#16,64,20,20
        return activated


#@ARCH_REGISTRY.register()
class SSFlow(nn.Module):
    def __init__(self, channels=[32, 64, 128, 128], connection=False):
        super(SSFlow, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        self.connection = connection
        self.E_block1 = nn.Sequential(
            nn.Conv2d(32, ch1, 3, stride=1, padding=1), nn.PReLU(),
            BasicBlock_E(ch1, ch2, mode='down'))
        self.E_block2 = BasicBlock_E(ch2, ch3, mode='down')
        self.E_block3 = BasicBlock_E(ch3, ch4, mode='down')

        self.side_out = nn.Conv2d(ch4, 3, 3, stride=1, padding=1)

        self.M_block1 = BasicBlock_E(ch4, ch4)
        self.M_block2 = BasicBlock_E(ch4, ch4)

        # dynamic filter
        ks_2d = 5
        self.conv_fac_k3 = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch4, ch4* ks_2d**2, 1, stride=1))

        self.conv_fac_k2 = nn.Sequential(
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch3, ch3* ks_2d**2, 1, stride=1))

        self.conv_fac_k1 = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(ch2, ch2* ks_2d**2, 1, stride=1))

        self.kconv_deblur = KernelConv2D(ksize=ks_2d, act=True)   #

        # curve
        self.curve_n = 3
        self.conv_1c = CurveCALayer(ch2, self.curve_n)
        self.conv_2c = CurveCALayer(ch3, self.curve_n)
        self.conv_3c = CurveCALayer(ch4, self.curve_n)

        self.PPM1 = PPM(ch2, ch2//4, bins=(1,2,3,6))
        self.PPM2 = PPM(ch3, ch3//4, bins=(1,2,3,6))
        self.PPM3 = PPM(ch4, ch4//4, bins=(1,2,3,6))

        self.D_block3 = BasicBlock_D_2Res(ch4, ch4)
        self.D_block2 = BasicBlock_D_2Res(ch4, ch3, mode='up')
        self.D_block1 = BasicBlock_D_2Res(ch3, ch2, mode='up')
        self.D_block0 = nn.Sequential(
            BasicBlock_D_2Res(ch2, ch1, mode='up'),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=1))

        self.conv22 = nn.Sequential(
            single_conv(128, 64),
        )
        self.conv33 = nn.Sequential(
            single_conv(128, 64),
        )
        # self.attn = FSAS(128, bias=False)
        # self.norm1 = LayerNorm(128, LayerNorm_type='WithBias')
        self.attn1 = FSAS(64, bias=False)
        self.norm1 = LayerNorm(64, LayerNorm_type='WithBias')
        self.attn2 = FSAS(128, bias=False)
        self.norm2 = LayerNorm(128, LayerNorm_type='WithBias')

        self.attn3 = FSAS(128, bias=False)
        self.norm3 = LayerNorm(128, LayerNorm_type='WithBias')
        self.OrthoNet=OrthoNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        self.BasicBlock=BasicBlock(inplanes=64, planes=64, height=64, stride=1, downsample=None)
    def forward(self, x, side_loss=False):
        # Dncoder
        e_feat1 = self.E_block1(x) #64 1/2  #down
        e_feat1 = self.BasicBlock(e_feat1)#

        e_feat1 = self.PPM1(e_feat1)


        e_feat2 = self.E_block2(e_feat1) #128 1/4
        e_feat2 = self.PPM2(e_feat2)


        e_feat3 = self.E_block3(e_feat2) #256 1/8
        e_feat3 = self.PPM3(e_feat3)


        if side_loss:
            out_side = self.side_out(e_feat3)     #

        # Mid
        m_feat = self.M_block1(e_feat3)#
        m_feat = self.M_block2(m_feat)#

        # Decoder
        d_feat3 = self.D_block3(m_feat) #256 1/8   #
        kernel_3  = self.conv_fac_k3(e_feat3)
        d_feat3 = self.kconv_deblur(d_feat3, kernel_3)  #
        if self.connection:
            d_feat3 = d_feat3 + e_feat3

        d_feat2 = self.D_block2(d_feat3)  #128 1/4    #2
        kernel_2  = self.conv_fac_k2(e_feat2)
        d_feat2 = self.kconv_deblur(d_feat2, kernel_2)
        if self.connection:
            d_feat2 = d_feat2 + e_feat2

        d_feat1 = self.D_block1(d_feat2)  #64 1/2   #
        kernel_1  = self.conv_fac_k1(e_feat1)
        d_feat1 = self.kconv_deblur(d_feat1, kernel_1)
        if self.connection:
            d_feat1 = d_feat1 + e_feat1

        out = self.D_block0(d_feat1)      #

        if side_loss:
            #return out_side, out
            return out,d_feat1,self.conv22(d_feat2),self.conv33(d_feat3)
        else:
            return out
