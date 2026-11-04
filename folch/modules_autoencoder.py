import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image #pillow library used for handling images
from torchvision import transforms
import numpy as np

from astropy.io import fits

class ConvBlock(nn.Module):
   def __init__(self, in_channels, out_channels): #input shape (in_channels, 60, 60)
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2) #output shape (out_channels, 30, 30)
   def forward(self, x):
    x = self.conv(x)
    x = F.relu(x)

    return x

class ConvTBlock(nn.Module):
   def __init__(self, in_channels, out_channels):
      super().__init__()
      self.convT = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
   def forward(self, x):
      x = self.convT(x)
      x = F.relu(x)

      return x
    

class autoenc(nn.Module):
    def __init__(self, in_channels):  #(1, 60, 60)
        super().__init__()
        #encoder
        self.conv1 = ConvBlock(in_channels, 32) #(32, 30, 30)
        self.conv2 = ConvBlock(32, 64) #(64, 15, 15)
        self.conv3 = ConvBlock(64, 128) #(128, 8, 8)
        self.conv4 = ConvBlock(128, 256) #(256, 4, 4)

        #bottleneck
        self.bottleneck = ConvBlock(256, 512) #(512, 2, 2)

        #decoder
        self.convT1 = ConvTBlock(512, 256) #(256, 4, 4)
        self.convT2 = ConvTBlock(256, 128) #(128, 8, 8)
        self.convT3 = ConvTBlock(128, 64) #(64, 16, 16)
        self.convT4 = ConvTBlock(64, 32) #(32, 32, 32)

        self.output = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=3, output_padding=1) #(1, 60, 60) 
       

    def forward(self, x):
       x1 = self.conv1(x)
       x2 = self.conv2(x1)
       x3 = self.conv3(x2)
       x4 = self.conv4(x3)

       latent = self.bottleneck(x4)

       d4 = self.convT1(latent)
       d3 = self.convT2(d4)
       d2 = self.convT3(d3)
       d1 = self.convT4(d2)

       output = F.sigmoid(self.output(d1))

       layers = [x1, x2, x3, x4, d4, d3, d2, d1]

       return latent, output, layers


class FizeauDataset(Dataset):
   def __init__(self, files, log=False): #files are .fits
      super().__init__()
      self.files = files
      self.log = log

   def __len__(self):
      return len(self.files)

   def rescale(self, tens, lower, upper):
      # log stretch first: these PSFs span many orders of magnitude, and
      # log10 of a value already scaled into [0, 1] is -inf at the minimum
      if self.log:
         tens = torch.log10(tens.clamp(min=0) + 1e-9)

      mx = tens.max()
      mn = tens.min()

      return (tens - mn) / (mx - mn + 1e-9) * (upper - lower) + lower 

   def __getitem__(self, index):
      img, hdr = fits.getdata(self.files[index], header=True)

      opd = torch.tensor(float(hdr["OPD"]), dtype=torch.float32)

      img = np.astype(img, np.float32)
      img = torch.from_numpy(img)
      img = torch.unsqueeze(img, dim=0)
      img = self.rescale(img, 0, 1)

      return img , opd
    

    
    
