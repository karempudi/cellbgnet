"""
A Convolutional Variational Autoencoder
"""
#https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=3, imsize=(28, 28)):
        super(VAE, self).__init__()
        self.featureDim = featureDim
        print(self.featureDim)
        self.imsize = imsize
        print(self.imsize)
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encConv3 = nn.Conv2d(32, 64, 5)
        self.encFC1 = nn.Linear(self.featureDim, zDim)
        self.encFC2 = nn.Linear(self.featureDim, zDim)
        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, self.featureDim)
        self.decConv1 = nn.ConvTranspose2d(64, 32, 5)
        self.decConv2 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv3 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = x.view(-1, self.featureDim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 64, self.imsize[0]-12, self.imsize[1]-12)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))

        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z
    

class VAE_CNN_version_2(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=3, imsize=(64, 64)):
        super(VAE_CNN_version_2, self).__init__()
        self.featureDim = featureDim
        print(self.featureDim)
        self.imsize = imsize
        print(self.imsize)
        c = 1
        out_channels = 16       
        self.encBn1 = nn.BatchNorm2d(c)
        self.encConv1 = nn.Conv2d(c, out_channels, kernel_size=3, stride=2, padding=1)  # 16, 32, 32
        self.encBn2 = nn.BatchNorm2d(out_channels)
        self.encConv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)  # 32, 16, 16
        self.encBn3 = nn.BatchNorm2d(out_channels*2)
        self.encConv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)  # 64, 8, 8
        self.encBn4 = nn.BatchNorm2d(out_channels*4)

        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder


        self.encFC1 = nn.Linear(out_channels*4*8*8, zDim)
        self.encFC2 = nn.Linear(out_channels*4*8*8, zDim)
        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, out_channels*4*8*8)
        self.decConv1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1) # 32, 16, 16
        self.decBn1 = nn.BatchNorm2d(out_channels*2)
        self.decConv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        self.decBn2 = nn.BatchNorm2d(out_channels)
        self.decConv3 = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        

        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        identity = x 
        out = self.encBn1(x)
        out = self.encConv1(out)
        out = F.leaky_relu(self.encBn2(out))
        out = self.encConv2(out)
        out = F.leaky_relu(self.encBn3(out))
        out = self.encConv3(out)

        out = self.encBn4(out)
        out = out.view(-1, 16*(2**2)*8*8)
        identity = identity.view(-1, 16*(2**2)*8*8)
        out += identity
        out = F.leaky_relu(out)   

        mu = self.encFC1(out)
        logVar = self.encFC2(out)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.leaky_relu(self.decFC1(z))
        x = x.view(-1, 64, 8, 8)

        identity = x 
        out = self.decConv1(x)
        out = F.leaky_relu(self.decBn1(out))

        out = self.decConv2(out)
        out = F.leaky_relu(self.decBn2(out))
        out = self.decConv3(out)
        identity = identity.view(out.shape)
        out += identity
        out = torch.sigmoid(out)   

        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z


class VAE_CNN_version_2_32x32(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=1, imsize=(32, 32)):
        super(VAE_CNN_version_2_32x32, self).__init__()
        self.featureDim = featureDim
        self.imsize = imsize
        c = 1
        out_channels = 16       
        self.encBn1 = nn.BatchNorm2d(c)
        self.encConv1 = nn.Conv2d(c, out_channels, kernel_size=3, stride=2, padding=1)  # 16, 16, 16
        self.encBn2 = nn.BatchNorm2d(out_channels)
        self.encConv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)  # 32, 8, 8
        self.encBn3 = nn.BatchNorm2d(out_channels*2)
        self.encConv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)  # 64, 4, 4
        self.encBn4 = nn.BatchNorm2d(out_channels*4)

        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder


        self.encFC1 = nn.Linear(out_channels*4*4*4, zDim)
        self.encFC2 = nn.Linear(out_channels*4*4*4, zDim)
        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, out_channels*4*4*4) # 64, 4, 4
        self.decConv1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1) # 32, 8, 8
        self.decBn1 = nn.BatchNorm2d(out_channels*2)
        self.decConv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 16, 16
        self.decBn2 = nn.BatchNorm2d(out_channels)
        self.decConv3 = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 1, 32, 32
        

        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        identity = x 
        out = self.encBn1(x)
        out = self.encConv1(out)
        out = F.leaky_relu(self.encBn2(out))
        out = self.encConv2(out)
        out = F.leaky_relu(self.encBn3(out))
        out = self.encConv3(out)

        out = self.encBn4(out)

        out = out.view(-1, 16*4*4*4)
        identity = identity.view(-1, 16*4*4*4)
        out += identity
        out = F.leaky_relu(out)   

        mu = self.encFC1(out)
        logVar = self.encFC2(out)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.leaky_relu(self.decFC1(z))
        x = x.view(-1, 64, 4, 4)

        identity = x 
        out = self.decConv1(x)
        out = F.leaky_relu(self.decBn1(out))

        out = self.decConv2(out)
        out = F.leaky_relu(self.decBn2(out))
        out = self.decConv3(out)
        identity = identity.view(out.shape)
        out += identity
        out = torch.sigmoid(out)   

        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z

class VAE_CNN_version_2_24x24(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=1, imsize=(20, 20)):
        super(VAE_CNN_version_2_24x24, self).__init__()
        self.featureDim = featureDim
        self.imsize = imsize
        c = 1
        out_channels = 16       
        self.encBn1 = nn.BatchNorm2d(c)
        self.encConv1 = nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1)  # 16, 16, 16
        self.encBn2 = nn.BatchNorm2d(out_channels)
        self.encConv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)  # 32, 8, 8
        self.encBn3 = nn.BatchNorm2d(out_channels*2)
        self.encConv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)  # 64, 4, 4
        self.encBn4 = nn.BatchNorm2d(out_channels*4)

        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder


        self.encFC1 = nn.Linear(out_channels*4*6*6, zDim)
        self.encFC2 = nn.Linear(out_channels*4*6*6, zDim)
        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, out_channels*4*6*6) # 64, 4, 4
        self.decConv1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1) # 32, 4, 4
        self.decBn1 = nn.BatchNorm2d(out_channels*2)
        self.decConv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 8, 8
        self.decBn2 = nn.BatchNorm2d(out_channels)
        self.decConv3 = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=1, padding=1, output_padding=0) # 1, 16, 16
        

        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        #identity = x 
        out = self.encBn1(x)
        out = self.encConv1(out)
        out = F.leaky_relu(self.encBn2(out))
        out = self.encConv2(out)
        out = F.leaky_relu(self.encBn3(out))
        out = self.encConv3(out)

        out = self.encBn4(out)
        #print(out.shape)
        out = out.view(-1, 16*4*6*6)
        #identity = identity.view(-1, 16*4*2*2)
        #out += identity
        out = F.leaky_relu(out)   

        mu = self.encFC1(out)
        logVar = self.encFC2(out)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.leaky_relu(self.decFC1(z))
        x = x.view(-1, 64, 6, 6)

        #identity = x 
        out = self.decConv1(x)
        out = F.leaky_relu(self.decBn1(out))
        #print(out.shape)
        out = self.decConv2(out)
        #print(out.shape)
        out = F.leaky_relu(self.decBn2(out))
        out = self.decConv3(out)
        #print(out.shape)
        #identity = identity.view(out.shape)
        #out += identity
        out = torch.sigmoid(out)   

        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z


class VAE_CNN_version_3(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=3, imsize=(64, 64)):
        super(VAE_CNN_version_2, self).__init__()
        self.featureDim = featureDim
        print(self.featureDim)
        self.imsize = imsize
        print(self.imsize)
        c = 1
        out_channels = 16       
        self.encBn1 = nn.BatchNorm2d(c)
        self.encConv1 = nn.Conv2d(c, out_channels, kernel_size=3, stride=2, padding=1)  # 16, 32, 32
        self.encBn2 = nn.BatchNorm2d(out_channels)
        self.encConv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)  # 32, 16, 16
        self.encBn3 = nn.BatchNorm2d(out_channels*2)
        self.encConv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)  # 64, 8, 8
        self.encBn4 = nn.BatchNorm2d(out_channels*4)
        
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder


        self.encFC1 = nn.Linear(out_channels*4*8*8, zDim)
        self.encFC2 = nn.Linear(out_channels*4*8*8, zDim)
        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, out_channels*4*8*8)
        self.decConv1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1) # 32, 16, 16
        self.decBn1 = nn.BatchNorm2d(out_channels*2)
        self.decConv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        self.decBn2 = nn.BatchNorm2d(out_channels)
        self.decConv3 = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        

        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        out = self.encBn1(x)
        out = self.encConv1(out)
        out = F.leaky_relu(self.encBn2(out))
        out = self.encConv2(out)
        out = F.leaky_relu(self.encBn3(out))
        out = self.encConv3(out)

        out = self.encBn4(out)
        out = out.view(-1, 16*(2**2)*8*8)
        identity = identity.view(-1, 16*(2**2)*8*8)
        out = F.leaky_relu(out)   

        mu = self.encFC1(out)
        logVar = self.encFC2(out)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.leaky_relu(self.decFC1(z))
        x = x.view(-1, 64, 8, 8)

        out = self.decConv1(x)
        out = F.leaky_relu(self.decBn1(out))

        out = self.decConv2(out)
        out = F.leaky_relu(self.decBn2(out))
        out = self.decConv3(out)
        identity = identity.view(out.shape)
        out = torch.sigmoid(out)   

        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z


class VAE_CNN_z_to_xyz(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*52*52, zDim=3, imsize=(64, 64)):
        super(VAE_CNN_version_2, self).__init__()
        self.featureDim = featureDim
        print(self.featureDim)
        self.imsize = imsize
        print(self.imsize)
        c = 1
        out_channels = 16       
        self.encBn1 = nn.BatchNorm2d(c)
        self.encConv1 = nn.Conv2d(c, out_channels, kernel_size=3, stride=2, padding=1)  # 16, 32, 32
        self.encBn2 = nn.BatchNorm2d(out_channels)
        self.encConv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1)  # 32, 16, 16
        self.encBn3 = nn.BatchNorm2d(out_channels*2)
        self.encConv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1)  # 64, 8, 8
        self.encBn4 = nn.BatchNorm2d(out_channels*4)

        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder


        self.encFC1z = nn.Linear(out_channels*4*8*8, 1)
        self.encFC2z = nn.Linear(out_channels*4*8*8, 1)

        self.encFC1xy = nn.Linear(out_channels*4*8*8, 2)
        self.encFC2xy = nn.Linear(out_channels*4*8*8, 2)

        
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1z = nn.Linear(1, out_channels*4*8*8)

        self.decFC1xy = nn.Linear(2, out_channels*4*8*8)


        self.decConv1 = nn.ConvTranspose2d(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1) # 32, 16, 16
        self.decBn1 = nn.BatchNorm2d(out_channels*2)
        self.decConv2 = nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        self.decBn2 = nn.BatchNorm2d(out_channels)
        self.decConv3 = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 16, 32, 32
        

        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        identity = x 
        out = self.encBn1(x)
        out = self.encConv1(out)
        out = F.leaky_relu(self.encBn2(out))
        out = self.encConv2(out)
        out = F.leaky_relu(self.encBn3(out))
        out = self.encConv3(out)

        out = self.encBn4(out)
        out = out.view(-1, 16*(2**2)*8*8)
        identity = identity.view(-1, 16*(2**2)*8*8)
        out += identity
        out = F.leaky_relu(out)   

        muz = self.encFC1z(out)
        logVarz = self.encFC2z(out)
        muxy = self.encFC1xy(out)
        logVarxy = self.encFC2xy(out)

        mu = torch.cat((muz, muxy), 1)
        logVar = torch.cat((logVarz, logVarxy), 1)

        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input

        

        x = self.decFC1z(z[0])
        x = F.leaky_relu(self.decFC1xy(z[1:])+x)

        x = x.view(-1, 64, 8, 8)

        identity = x 
        out = self.decConv1(x)
        out = F.leaky_relu(self.decBn1(out))

        out = self.decConv2(out)
        out = F.leaky_relu(self.decBn2(out))
        out = self.decConv3(out)
        identity = identity.view(out.shape)
        out += identity
        out = torch.sigmoid(out)   

        return out

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar, z