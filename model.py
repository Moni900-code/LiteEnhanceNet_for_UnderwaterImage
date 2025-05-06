import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility Functions
def gelu(x):
    return x * torch.sigmoid(1.702 * x)  # Approximation of GELU

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            HardSigmoid(inplace=True)
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        channel_attn = self.channel_gate(x)
        x = x * channel_attn
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1)))
        x = x * spatial_attn
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, use_cbam=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.cbam(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, groups=out_channels)  # Depthwise
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)  # Pointwise
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + self.shortcut(residual)
        x = F.gelu(x)
        return x

class EnhancedUnderwaterNet(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedUnderwaterNet, self).__init__()
        self.input_conv = ConvBlock(3, 32, kernel_size=3, stride=1)

        # Multi-scale feature extraction
        self.stage1 = ResidualBlock(32, 64, stride=1)
        self.stage2 = ResidualBlock(64, 128, stride=2)  # Downsample
        self.stage3 = ResidualBlock(128, 256, stride=2)  # Downsample
        self.stage4 = ResidualBlock(256, 256, stride=1)

        # Upsampling and feature fusion
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.fuse1 = ConvBlock(256, 128)  # Concat with stage2
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.fuse2 = ConvBlock(128, 64)   # Concat with stage1

        # Output layer
        self.output = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        self.final_act = nn.Tanh()  # Ensure output is in [-1, 1] for image enhancement

    def forward(self, x):
        # Encoder
        x1 = self.input_conv(x)      # 32 channels
        x2 = self.stage1(x1)         # 64 channels
        x3 = self.stage2(x2)         # 128 channels, downsampled
        x4 = self.stage3(x3)         # 256 channels, downsampled
        x5 = self.stage4(x4)         # 256 channels

        # Decoder with skip connections
        x = self.up1(x5)             # Upsample to 128 channels
        x = torch.cat([x, x3], dim=1)  # Skip connection from stage2
        x = self.fuse1(x)            # Fuse to 128 channels
        x = self.up2(x)              # Upsample to 64 channels
        x = torch.cat([x, x2], dim=1)  # Skip connection from stage1
        x = self.fuse2(x)            # Fuse to 64 channels

        # Output
        x = self.output(x)
        x = self.final_act(x)
        return x

# Example usage
if __name__ == "__main__":
    model = EnhancedUnderwaterNet(num_classes=3)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)  # Should be [1, 3, 256, 256]
