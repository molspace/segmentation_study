import torch
import torch.nn as nn
from configs import n_classes

# UNetDownBlock: Conv + ReLU + Conv + ReLU + MaxPool
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        # x: a tensor of shape (batch_size, in_channels, layer_height, layer_width)
        out_before_pooling = self.convs(x)
        out = self.maxpool(out_before_pooling)
        return out, out_before_pooling


# UNetUpBlock: upsampling + concat + Conv + ReLU
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x, x_bridge):
        # x: a tensor of shape (batch_size, in_channels, layer_height // 2, layer_width // 2)
        # x_bridge: a tensor of shape (batch_size, in_channels, layer_height, layer_width)
        x_up = self.upsample(x)
        x_concat = torch.cat([x_up, x_bridge], dim=1)
        out = self.convs(x_concat)
        return out


# Let's make a UNet with 5 levels and 64, 128, 256, 256, 256 channels
# (n_base_channels=64)
class UNet(nn.Module):
    def __init__(self, n_base_channels=64):
        super().__init__()
        self.down_blocks = nn.ModuleList([
            UNetDownBlock(3, n_base_channels),
            UNetDownBlock(n_base_channels, n_base_channels * 2),
            UNetDownBlock(n_base_channels * 2, n_base_channels * 4),
            UNetDownBlock(n_base_channels * 4, n_base_channels * 4),
            UNetDownBlock(n_base_channels * 4, n_base_channels * 4)
        ])
        self.up_blocks = nn.ModuleList([
            UNetUpBlock(n_base_channels * 4, n_base_channels * 4),
            UNetUpBlock(n_base_channels * 4, n_base_channels * 2),
            UNetUpBlock(n_base_channels * 2, n_base_channels),
            UNetUpBlock(n_base_channels, n_base_channels),
        ])
        self.final_block = nn.Sequential(
            #  nn.Conv2d(n_base_channels, 3, kernel_size=1, padding=0),
            nn.Conv2d(n_base_channels, n_classes, kernel_size=1, padding=0),
        )


    def forward(self, x):
        out = x
        outputs_before_pooling = []
        for i, block in enumerate(self.down_blocks):
            out, before_pooling = block(out)
            outputs_before_pooling.append(before_pooling)
            # easy mistake can be made here: for last layer, we need to save out_before_pooling
        out = before_pooling

        # now outputs_before_pooling = [block1_before_pooling, ..., block5_before_pooling]        
        for i, block in enumerate(self.up_blocks):    # NB: it's easier to understand when using counter (i=3, etc.)
            out = block(out, outputs_before_pooling[-i - 2])
        out = self.final_block(out)
        return out