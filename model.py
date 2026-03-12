import torch
import torch.nn as nn
import math

# === 修改 1: RDB 内部强制使用 reflect padding ===
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        
        def make_conv(in_channels, out_channels):
            return nn.Sequential(
                # [关键] padding_mode='reflect' 消除棋盘格和边缘伪影
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv1 = make_conv(channels, growth_rate)
        self.conv2 = make_conv(channels + growth_rate, growth_rate)
        self.conv3 = make_conv(channels + 2 * growth_rate, growth_rate)
        
        # 融合层也需要 reflect padding
        self.local_fuse = nn.Conv2d(channels + 3 * growth_rate, channels, kernel_size=1, padding=0) 
        # 1x1 卷积不需要 padding，所以保持默认

    def forward(self, x):
        identity = x 
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat((x, out1), 1))
        out3 = self.conv3(torch.cat((x, out1, out2), 1))
        out_fused = self.local_fuse(torch.cat((x, out1, out2, out3), 1))
        return out_fused + identity

class Generator(nn.Module):
    def __init__(self, scale_factor=4, num_residual_blocks=16, noise_std=0.01):
        super(Generator, self).__init__()
        self.noise_std = noise_std
        
        # [关键] 入口层 Reflect Padding
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4, padding_mode='reflect')
        self.prelu = nn.PReLU()
        
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualDenseBlock(channels=64)) 
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # [关键] 中间层 Reflect Padding
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(64)
        
        # === 修改 2: 无伪影上采样结构 ===
        upsampling = []
        for _ in range(int(math.log(scale_factor, 2))):
            # 1. 最近邻插值 (对于 Ising 这种二值数据，这是最好的选择)
            upsampling.append(nn.Upsample(scale_factor=2, mode='nearest'))
            # 2. 卷积层 (Reflect Padding)
            upsampling.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect'))
            # 3. 激活
            upsampling.append(nn.PReLU())
            
        self.upsampling = nn.Sequential(*upsampling)
        
        # [关键] 输出层 Reflect Padding + Tanh
        self.conv3 = nn.Conv2d(64, 1, kernel_size=9, padding=4, padding_mode='reflect')
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x = self.res_blocks(x1)

        if self.training:
            # 噪声注入，增加随机性防止坍塌
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        x = self.bn2(self.conv2(x))
        x = x + x1
        x = self.upsampling(x)
        x = self.conv3(x)
        # 输出范围 [-1, 1]
        return self.tanh(x)

# === 修改 3: 移除 Spectral Norm ===
# 之前的 Spectral Norm 限制太死，导致生成器一旦产生的图像全黑，
# 判别器也能达到纳什均衡，导致无法跳出局部最优。回归标准 D 结构。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = 1
        layers.extend(discriminator_block(in_filters, 64, first_block=True))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512)) # 加深一点网络

        self.model = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_out = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.conv_out(x)
        return self.sigmoid(x).view(-1)