import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-6, groups=1, demodulation=True):
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

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W] 
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
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
        self.norm = ChannelNorm(channels)
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

class ToRGB(nn.Module):
    def __init__(self, channels, style_dim):
        super(ToRGB, self).__init__()
        self.a = nn.Linear(style_dim, 3)
        self.c = Conv2dMod(channels, 3, 1, demodulation=False)

    def forward(self, x, y):
        return self.c(x, self.a(y))

class FromRGB(nn.Module):
    def __init__(self, channels):
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, channels, 1, 1, 0)
        self.norm = ChannelNorm(channels)

    def forward(self, x):
        return self.norm(self.conv(x))

class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x

class EqualLinear(nn.Module):
    def __init__(self, input_dim, output_dim, lr_mul=0.1):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.lr_mul = lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, self.bias *  self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[nn.Sequential(EqualLinear(style_dim, style_dim), nn.GELU()) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(style_dim)
    def forward(self, x):
        return self.norm(self.seq(x))

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim, num_layers=2, upscale=True):
        super(GeneratorBlock, self).__init__()
        self.upscale = nn.Upsample(scale_factor=2) if upscale else nn.Identity()
        self.layers = nn.ModuleList([ConvNeXtModBlock(input_channels, style_dim) for _ in range(num_layers)])
        self.conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.to_rgb = ToRGB(output_channels, style_dim)
    
    def forward(self, x, y):
        x = self.upscale(x)
        for l in self.layers:
            x = l(x, y)
        x = self.conv(x)
        rgb = self.to_rgb(x, y)
        return x, rgb

class Generator(nn.Module):
    def __init__(self, initial_channels=512, style_dim=512, num_layers_per_block=2):
        super(Generator, self).__init__()
        self.initial_param = nn.Parameter(torch.randn(1, initial_channels, 8, 8))
        self.last_channels = initial_channels
        self.style_dim = style_dim
        self.num_layers_per_block = num_layers_per_block
        self.layers = nn.ModuleList([])
        self.upscale = nn.Sequential(nn.Upsample(scale_factor=2), Blur())

        self.add_layer(upscale=False)

    def forward(self, styles):
        if type(styles) != list:
            styles = [styles] * len(self.layers)
        rgb_out = None
        x = self.initial_param.expand(styles[0].shape[0], -1, -1, -1)
        for i, l in enumerate(self.layers):
            x, rgb = l(x, styles[i])
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = self.upscale(rgb_out) + rgb
        return rgb_out

    def add_layer(self, upscale=True):
        ch = self.last_channels // 2
        if ch < 16:
            ch = 16
        self.layers.append(GeneratorBlock(self.last_channels, ch, self.style_dim, num_layers=self.num_layers_per_block, upscale=upscale))
        self.last_channels = ch

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2, downscale=True):
        super(DiscriminatorBlock, self).__init__()
        self.from_rgb = FromRGB(input_channels)
        self.conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.layers = nn.ModuleList([ConvNeXtBlock(output_channels) for _ in range(num_layers)])
        self.downscale = nn.Sequential(nn.AvgPool2d(kernel_size=2), ChannelNorm(output_channels)) if downscale else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        for l in self.layers:
            x = l(x)
        x = self.downscale(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, initial_channels=512, num_layers_per_block=2):
        super(Discriminator, self).__init__()
        self.last_channels = initial_channels
        self.num_layers_per_block = num_layers_per_block
        self.pool8x = nn.AvgPool2d(kernel_size=8)
        self.layers = nn.ModuleList([])
        self.downscale = nn.Sequential(Blur(), nn.AvgPool2d(kernel_size=2))
        self.ffn = nn.Sequential(
                nn.Linear(initial_channels + 1, initial_channels//4),
                nn.GELU(),
                nn.Linear(initial_channels //4, 1))
        self.add_layer(False)

    def forward(self, rgb):
        x = self.layers[0].from_rgb(rgb)
        for i, l in enumerate(self.layers):
            if i == 1:
                x += self.layers[1].from_rgb(self.downscale(rgb))
            x = l(x)
        x = self.pool8x(x)
        mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, mb_std], dim=1)
        x = self.ffn(x)
        return x
        
    def add_layer(self, downscale=True):
        ch = self.last_channels // 2
        if ch < 16:
            ch = 16
        self.layers.insert(0, DiscriminatorBlock(ch, self.last_channels, self.num_layers_per_block, downscale))
        self.last_channels = ch


