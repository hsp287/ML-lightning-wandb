import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_shape=(28,28), in_channels=1, out_channels=10, k=3):
        super(CNN, self).__init__()

        H, W = in_shape
        H_, W_ = H//4, W//4   # overall downsampling by 4
        
        # will downsample by 2
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*4, k, padding=k//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # will downsample by 2
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*16, k, padding=k//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.fc1 = nn.Linear(in_channels*16*H_*W_, 128)
        self.fc2 = nn.Linear(128, out_channels)
        #self.act = nn.Softmax()

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc1(x)
        out = self.fc2(x)
        return out