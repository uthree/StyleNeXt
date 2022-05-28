import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv2dMod(nn.Module):
    """Some Information about Conv2dMod"""
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-4, groups=1, demodulation=True):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels // groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.demodulation = demodulation
        self.groups = groups

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W) 
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape
        
        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w2 * (w1 + 1)

        # demodulate
        if self.demodulation:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N, *ws)
        
        
        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        
        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N * self.groups)
        x = x.reshape(N, self.output_channels, H, W)

        return x

class ConvNeXtModBlock(nn.Module):
    def __init__(self, channels, style_dim, dim_ffn=None, kernel_size=7):
        super(ConvNeXtModBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.a1 = nn.Linear(style_dim, channels)
        self.c1 = Conv2dMod(channels, channels, kernel_size=kernel_size, groups=channels)
        self.a2 = nn.Linear(style_dim, dim_ffn)
        self.c2 = Conv2dMod(channels, dim_ffn, kernel_size=1)
        self.gelu = nn.GELU()
        self.a3 = nn.Linear(style_dim, channels)
        self.c3 = Conv2dMod(dim_ffn, channels, kernel_size=1)

    def forward(self, x, y):
        res = x
        x = self.c1(x, self.a1(y))
        x = self.c2(x, self.a2(y))
        x = self.gelu(x)
        x = self.c3(x, self.a3(y))
        return x + res

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim_ffn=None, kernel_size=7):
        super(ConvNeXtBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.c1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, padding_mode='replicate', groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.c2 = nn.Conv2d(channels, dim_ffn, 1, 1, 0)
        self.gelu = nn.GELU()
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)
    
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.gelu(x)
        x = self.c3(x)
        return x + res

