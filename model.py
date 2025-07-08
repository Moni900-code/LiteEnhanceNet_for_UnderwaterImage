import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from ptflops import get_model_complexity_info

# -------------------------
# ConvBlock (CBAM Removed)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs = nn.Hardswish()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.hs(x)
        x = self.pw(x)
        x = self.bn2(x)
        return x

# -------------------------
# Lightweight Color Feature Extractor
# -------------------------
class ColorFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ColorFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------
# Color Recovery Module (CRM)
# -------------------------
class ColorRecoveryModule(nn.Module):
    def __init__(self, in_channels):
        super(ColorRecoveryModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, content_features, color_features):
        D = -content_features - color_features
        M = content_features * color_features
        L = 2 * torch.sigmoid(D) * torch.tanh(M)

        sigmoid_D = torch.sigmoid(D)
        sigmoid_D = torch.clamp(sigmoid_D, 0, 0.5)

        L = torch.clamp(L, 0, 1)

        output_features = []
        current_color = color_features

        for _ in range(4):
            F_i = L * current_color + content_features
            output_features.append(F_i)
            current_color = self.conv1x1(F_i)
            current_color = F.relu(current_color)

        final_output = torch.mean(torch.stack(output_features), dim=0)
        return final_output

# -------------------------
# Mynet without CBAM
# -------------------------
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=False)
        self.bn_input = nn.BatchNorm2d(16)
        self.hs_input = nn.Hardswish()

        self.block1 = ConvBlock(16, 32, stride=1)
        self.block2 = ConvBlock(32, 64, stride=1)
        self.block3 = ConvBlock(80, 32, stride=1)  # ⛔ CBAM removed

        self.color_extractor = ColorFeatureExtractor(in_channels=3, out_channels=32)
        self.crm = ColorRecoveryModule(in_channels=32)

        self.output = nn.Conv2d(32, 3, kernel_size=1, stride=1)
        self.final_act = nn.Tanh()

    def forward(self, x, gt_color_source=None):
        color_input = gt_color_source if gt_color_source is not None else x
        color_features = self.color_extractor(color_input)

        x = self.input(x)
        x = self.bn_input(x)
        x = self.hs_input(x)

        x = self.block1(x)
        x = self.block2(x)
        x = torch.cat([x, torch.zeros_like(x)[:, :16, :, :]], dim=1)  # pad to 80 channels
        content_features = self.block3(x)

        color_features = F.interpolate(color_features, size=content_features.shape[2:], mode='bilinear', align_corners=False)
        x = self.crm(content_features, color_features)

        x = self.output(x)
        x = self.final_act(x)
        return x

# -------------------------
# Main: Summary + FLOPs
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mynet().to(device)

    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")

    print("\nModel Summary:")
    summary(model, input_size=(3, 224, 224))

    print("\nCalculating FLOPs:")
    with torch.cuda.device(0 if torch.cuda.is_available() else "cpu"):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"\nFLOPs: {macs}")
        print(f"Parameters: {params}")
