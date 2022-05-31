import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
import multiprocessing
import numpy as np
from PIL import Image

from dataset import ImageDataset

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
        self.norm = ChannelNorm(channels)
        self.a2 = nn.Linear(style_dim, dim_ffn)
        self.c2 = Conv2dMod(channels, dim_ffn, kernel_size=1)
        self.gelu = nn.GELU()
        self.a3 = nn.Linear(style_dim, channels)
        self.c3 = Conv2dMod(dim_ffn, channels, kernel_size=1)

    def forward(self, x, y):
        res = x
        x = self.c1(x, self.a1(y))
        x = self.norm(x)
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
        return F.linear(x, self.weight * self.lr_mul, self.bias * self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(*[nn.Sequential(EqualLinear(style_dim, style_dim), nn.GELU()) for _ in range(num_layers)])
        self.prenorm = nn.LayerNorm(style_dim)
    def forward(self, x):
        return self.seq(self.prenorm(x))

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim, num_layers=2, upscale=True):
        super(GeneratorBlock, self).__init__()
        self.upscale = nn.Sequential(nn.Upsample(scale_factor=2), ChannelNorm(input_channels)) if upscale else nn.Identity()
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
        self.initial_param = nn.Parameter(torch.ones(1, initial_channels, 8, 8))
        self.last_channels = initial_channels
        self.style_dim = style_dim
        self.num_layers_per_block = num_layers_per_block
        self.layers = nn.ModuleList([])
        self.upscale = nn.Sequential(nn.Upsample(scale_factor=2), Blur())
        self.alpha = 0

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
                rgb_out = self.upscale(rgb_out) + rgb * self.alpha
        rgb_out = torch.tanh(rgb_out)
        return rgb_out

    def add_layer(self, upscale=True):
        self.alpha = 0
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
                nn.Linear(initial_channels //4, 1))
        self.add_layer(False)
        self.alpha = 0

    def forward(self, rgb):
        x = self.layers[0].from_rgb(rgb)
        if len(self.layers) > 1:
            x = x * self.alpha
        for i, l in enumerate(self.layers):
            if i == 1:
                x += self.layers[1].from_rgb(self.downscale(rgb)) * (1 - self.alpha)
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
        self.alpha = 0

class GAN(nn.Module):
    def __init__(
            self,
            initial_channels=512,
            style_dim=512,
            max_resolution=512,
            num_layers_per_block=2
            ):
        super(GAN, self).__init__()
        self.style_dim = style_dim
        self.resolution = 8
        self.max_resolution = max_resolution
        self.mapping_network = MappingNetwork(style_dim)
        self.generator = Generator(initial_channels, style_dim, num_layers_per_block)
        self.discriminator = Discriminator(initial_channels, num_layers_per_block)

    def train_resolution(self, dataset, device=torch.device('cpu'), batch_size=1, augmentation=nn.Identity(), lr=1e-5, num_epoch=1, result_dir="./results/", model_path='./model.pt'):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        bar_epoch = tqdm(total=num_epoch * len(dataset), position=0)
        bar_batch = tqdm(total=len(dataset), position=1)
        bar_epoch.set_description(desc='[Loading]')
        bar_epoch.update(0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        T = augmentation
        M, G, D, = self.mapping_network, self.generator, self.discriminator
        M.to(device)
        G.to(device)
        D.to(device)

        opt_m = optim.RAdam(M.parameters(), lr=lr)
        opt_g = optim.RAdam(G.parameters(), lr=lr)
        opt_d = optim.RAdam(D.parameters(), lr=lr)
        
        for i in range(num_epoch):
            for j, images in enumerate(dataloader):
                images = images.to(device)
                N = images.shape[0]
                # caluclate alpha
                alpha = min(1.0, (bar_epoch.n / (bar_epoch.total / 2)))
                G.alpha = alpha
                D.alpha = alpha

                # train generator
                opt_g.zero_grad()
                opt_m.zero_grad()
                z1, z2 = torch.randn(N, self.style_dim, device=device), torch.randn(N, self.style_dim, device=device)
                w1, w2 = M(z1), M(z2)
                L = random.randint(0, len(self.generator.layers))
                styles = [w1] * L + [w2] * (len(self.generator.layers) - L)
                fake = G(styles)
                g_loss = -D(fake).mean()
                g_loss.backward()
                opt_g.step()
                opt_m.step()
                
                # train discriminator
                opt_d.zero_grad()
                fake = fake.detach()
                fake = T(fake)
                real = T(images)
                d_loss_f = -torch.minimum(-D(fake)-1, torch.zeros(N, 1, device=device)).mean()
                d_loss_r = -torch.minimum(D(real)-1, torch.zeros(N, 1, device=device)).mean()
                d_loss = d_loss_f + d_loss_r
                d_loss.backward()
                opt_d.step()
                tqdm.write(f"Batch: {j} G.Loss: {g_loss.item():.6f} D.Loss: {d_loss.item():.6f} Alpha: {alpha:.4f}")
                bar_batch.set_description(desc=f"[Batch {j}] G.Loss: {g_loss.item():.4f} D.Loss: {d_loss.item():.4f}")
                bar_epoch.set_description(desc=f"[Epoch {i}]")
                bar_batch.update(N)
                bar_epoch.update(N)
                if j % 1000 == 0:
                    # save result
                    file_name = f"{i}_{j}.jpg"
                    img = Image.fromarray((fake[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
                    img = img.resize((256, 256))
                    img.save(os.path.join(result_dir, file_name))
                    torch.save(self, model_path)
                    tqdm.write("Saved Model.")
            bar_batch.reset()
                    
    def train(self, pathes=[], num_epoch=1, batch_size=1, max_len=100000, model_path='./model.pt', device=torch.device('cpu'), lr=1e-5):
        ds = ImageDataset(pathes, size=8, max_len=max_len)
        while True:
            ds.set_size(self.resolution)
            div_bs = (self.resolution // 8)
            bs = batch_size // div_bs
            if bs < 4:
                bs = 4
            print(f"Training resolution: {self.resolution}x, batch size: {bs}")
            aug = transforms.RandomApply([transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.RandomRotation((-20, 20)),
                        transforms.RandomCrop((round(self.resolution * 0.8), round(self.resolution * 0.8)))], p=0.5),
                    transforms.Resize((self.resolution, self.resolution))
                    ])], p=0.5)
            self.train_resolution(ds, device, bs, aug, lr, num_epoch)
            if self.resolution >= self.max_resolution:
                print("Training Complete!")
                break
            self.resolution *= 2
            self.discriminator.add_layer()
            self.generator.add_layer()

    def generate_random_image(self, num_images, scale=1.0, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = []
        for i in range(num_images):
            with torch.no_grad():
                z1 = torch.randn(1, self.style_dim).to(device)
                z2 = torch.randn(1, self.style_dim).to(device)
                w1 = self.mapping_network(z1) * scale
                w2 = self.mapping_network(z2) * scale
                L = random.randint(1, len(self.generator.layers))
                style = [w1] * L + [w2] * (len(self.generator.layers)-L)
                image = self.generator(style)
                image = image.detach().cpu().numpy()
                images.append(image[0])
        return images

    def generate_random_image_to_directory(self, num_images, dir_path="./tests", scale=1.0):
        images = self.generate_random_image(num_images, scale=scale)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        print("Generating...")
        for i in tqdm(range(num_images)):
            img = Image.fromarray((images[i] * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img = img.resize((256, 256))
            img.save(os.path.join(dir_path, f"{i}.jpg"), mode='RGB')

