import torch.nn as nn
import torch


#class named DoubleConv interiting from nn.Module class 
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    #defining the constructor with the following structure as mid_channels = convolution followed by batch normalization followed by 
    #LeakyReLU followed by convolution followed by  batch normalization and then lastly the leakyReLU activation function 

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

     #function to perform the forward pass
    def forward(self, x):
        return self.double_conv(x)

#class named Down inheriting from nn.Module class 
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    #constructor with maxpool_conv = max pooling followed by the DoubleConv object as defined in the previous class 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    #function to perform the forward pass  for this module 
    def forward(self, x):
        return self.maxpool_conv(x)

#class named Up inheriting from nn.Module class
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    #function to perform the forward pass  for this module 
    #more details can be found in the UNet paper 
    def forward(self, x1, x2, x3=None):
        x1 = self.up(x1)
        # input is CHW
        if x3 != None:
            x = torch.cat([x3, x2, x1], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#class named OutConv inheriting from nn.Module class
class OutConv(nn.Module):
    #constructor with conv = convolutional layer
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    #function to perform the forward pass  for this module 
    def forward(self, x):
        return self.conv(x)


#class named UNet inheriting from nn.Module class
#defines the layers in the Hybrid UNet architecture and their order and forwards pass functions 
#more details on the architecture can be found in the medium article with the figure 
class HybridUNet(nn.Module):
    def __init__(self, 
                 model_cfg: dict,
                 device="cpu"):
        # n_channels, n_classes, bilinear=True
        super(HybridUNet, self).__init__()
        self.n_channels = model_cfg["input_channel"]
        self.n_classes = model_cfg["output_classes"]
        self.bilinear = model_cfg['bilinear']

        self.inc = DoubleConv(model_cfg["input_channel"], 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if model_cfg['bilinear'] else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1088, 512 // factor, model_cfg['bilinear'])
        self.up2 = Up(640, 256 // factor, model_cfg['bilinear'])
        self.up3 = Up(256, 128 // factor, model_cfg['bilinear'])
        self.up4 = Up(128, 64, model_cfg['bilinear'])
        self.outc = OutConv(64, model_cfg["output_classes"])

        self.down1_1 = nn.MaxPool2d(8)
        self.down2_2 = nn.MaxPool2d(2)


    def forward(self, x):
        x1 = self.inc(x)
        x1_1 = self.down1_1(x1)
        x2 = self.down1(x1)
        x2_2 = self.down2_2(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, x1_1)
        x = self.up2(x, x3, x2_2)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
from torchsummary import summary

if __name__ == "__main__":
    summary(HybridUNet(n_channels=3, n_classes=2), (3, 64, 64), device="cpu")