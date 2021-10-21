import torch
import torch.nn as nn

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBlock, self).__init__()
        if mid_channels == None:
            mid_channels = in_channels//4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels))
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else (lambda x:x)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x)) 
        x = x + self.down(res)
        return x

    
class Blocks(nn.Sequential):
    def __init__(self, block_args):
        super(Blocks, self).__init__()
        for i, channels in enumerate(block_args):
            if len(channels) != 2:
                in_c, mid_c, out_c = channels
                self.add_module(f'resblock-{i}', ResBlock(in_c, out_c, mid_channels=mid_c))
            else:
                in_c, out_c = channels
                self.add_module(f'resblock-{i}', ResBlock(in_c, out_c))
    
    
class Model(nn.Module):
    def __init__(self, block_args):
        super(Model, self).__init__()
        self.layer = Blocks(block_args)
        self.fc = nn.Linear(block_args[-1][-1], 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        
        x = self.layer(x)
        
        N, C, H, W = x.size()
        x = x.view(N, C, -1).mean(-1)
        
        x = self.fc(x)
        return self.softmax(x)