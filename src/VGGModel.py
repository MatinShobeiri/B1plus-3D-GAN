import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import functools


class VGG_encoder(nn.Module):
    def __init__(self, features=32, in_channels=1, maxpool=False, avgpool=False,):
        super(VGG_encoder, self).__init__()

        conv_stride = 1 if (maxpool + avgpool) else 2
        
        pooling = getattr(nn, 'MaxPool3d')
        if avgpool==True:
            pooling = getattr(nn, 'AvgPool3d')

        self.encoder1 = VGG_encoder._block(in_channels, features, name="enc1", stride=1)
        self.pool1 = pooling(kernel_size=2, stride=2) if (maxpool+avgpool) else Identity()    
        self.encoder2 = VGG_encoder._block(features, features * 2, name="enc2", stride=conv_stride)
        self.pool2 = pooling(kernel_size=2, stride=2) if (maxpool+avgpool) else Identity()    
        self.encoder3 = VGG_encoder._block(features * 2, features * 4, name="enc3", stride=conv_stride)
        self.pool3 = pooling(kernel_size=2, stride=2) if (maxpool+avgpool) else Identity()    
        self.encoder4 = VGG_encoder._block(features * 4, features * 8, name="enc4", stride=conv_stride)
        self.pool4 = pooling(kernel_size=2, stride=2) if (maxpool+avgpool) else Identity()    

        self.bottleneck = VGG_encoder._block(features * 8, features * 16, name="bottleneck", stride=conv_stride)

    def forward(self, x):
        x = self.pool1(self.encoder1(x))
        x = self.pool2(self.encoder2(x))
        x = self.pool3(self.encoder3(x))
        x = self.pool4(self.encoder4(x))
        return self.bottleneck(x)


    @staticmethod
    def _block(in_channels, features, name, stride=1):
        return nn.Sequential(
            OrderedDict([
                    (name + "conv1",
                        nn.Conv3d(in_channels=in_channels,out_channels=features,kernel_size=3,stride=stride,padding=1,bias=False,),),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2",
                        nn.Conv3d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False,),),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]))

    
class VGG_decoder(nn.Module):
    def __init__(self, features=32):
        super(VGG_decoder, self).__init__()

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = VGG_encoder._block(features * 8, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = VGG_encoder._block(features * 4, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = VGG_encoder._block(features * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = VGG_encoder._block(features, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.decoder4(self.upconv4(x))
        x = self.decoder3(self.upconv3(x))
        x = self.decoder2(self.upconv2(x))
        x = self.upconv1(x)
        x = self.decoder1(x)
        return self.sig(self.conv(x)) #torch.sigmoid(self.conv(x))


class Discriminator(nn.Module):
    def __init__(self, features=32, maxpool=False, avgpool=False):
        super(Discriminator, self).__init__()

        conv_stride = 1 if (maxpool + avgpool) else 2
        
        pooling = getattr(nn, 'MaxPool3d')
        if avgpool==True:
            pooling = getattr(nn, 'AvgPool3d')

        self.model = nn.Sequential(*[
        VGG_encoder._block(2, features, name="enc1", stride=1),
        # pooling(kernel_size=2, stride=2),
        VGG_encoder._block(features, features * 2, name="enc2", stride=conv_stride),
        # pooling(kernel_size=2, stride=2),
        VGG_encoder._block(features * 2, features * 4, name="enc3", stride=conv_stride),
        # pooling(kernel_size=2, stride=2),
        VGG_encoder._block(features * 4, features * 8, name="enc4-1", stride=conv_stride),
        # pooling(kernel_size=2, stride=2),
        VGG_encoder._block(features * 8, features * 8, name="enc4-2", stride=conv_stride),
        VGG_encoder._block(features * 8, features * 8, name="enc4-3", stride=conv_stride),
        VGG_encoder._block(features * 8, features * 16, name="bottleneck", stride=conv_stride),
        VGG_encoder._block(features * 16, 1, name="compressor"),
        ])

    def forward(self, x):
        x = self.model(x)
        x = nn.AvgPool3d(kernel_size=x.size()[2:])(x)
        x = F.sigmoid(x)
        return x.view(x.size()[0])
