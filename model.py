import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from ptflops import get_model_complexity_info

# -------------------------
# CBAM Module
# -------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, bias=False),
            nn.Hardsigmoid()
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        channel_att = self.channel_gate(x)
        x = x * channel_att
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_gate(spatial_input)
        spatial_att = torch.sigmoid(spatial_att)
        x = x * spatial_att
        return x

# -------------------------
# ConvBlock
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ConvBlock, self).__init__()
        self.use_cbam = use_cbam
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs = nn.Hardswish()
        if self.use_cbam:
            self.cbam = CBAM(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.hs(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.pw(x)
        x = self.bn2(x)
        return x

# -------------------------
# Lightweight Color Feature Extractor (for color histogram)
# -------------------------
class ColorFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ColorFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),  # Reduced channels to 16
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
        self.in_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, content_features, color_features):
        # Equation (4): D(x, y) = -Con(x,y) - Color(x,y)
        D = -content_features - color_features
        
        # Equation (5): M(x,y) = Con(x,y) * Color(x,y)
        M = content_features * color_features
        
        # Equation (6): L = 2 * sigmoid(D) * tanh(M)
        L = 2 * torch.sigmoid(D) * torch.tanh(M)
        
        # Ensure sigmoid(D) ∈ [0, 0.5]
        sigmoid_D = torch.sigmoid(D)
        sigmoid_D = torch.clamp(sigmoid_D, 0, 0.5)
        
        # Ensure L ∈ [0, 1]
        L = torch.clamp(L, 0, 1)
        
        # Equation (7): F_i, Color_{i+1} = L_i * Color_i + Con_i for i ∈ [1, 4]
        output_features = []
        current_color = color_features
        
        for i in range(4):
            F_i = L * current_color + content_features
            output_features.append(F_i)
            current_color = self.conv1x1(F_i)
            current_color = F.relu(current_color)
        
        final_output = torch.mean(torch.stack(output_features), dim=0)
        return final_output

# -------------------------
# Mynet with Lightweight CRM
# -------------------------
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=False)
        self.bn_input = nn.BatchNorm2d(16)
        self.hs_input = nn.Hardswish()
        
        self.block1 = ConvBlock(16, 32, stride=1)
        self.block2 = ConvBlock(32, 64, stride=1)
        self.block3 = ConvBlock(80, 32, stride=1, use_cbam=True)
        
        # Lightweight color feature extractor
        self.color_extractor = ColorFeatureExtractor(in_channels=3, out_channels=32)
        
        # CRM integration
        self.crm = ColorRecoveryModule(in_channels=32)
        
        self.output = nn.Conv2d(32, 3, kernel_size=1, stride=1)
        self.final_act = nn.Tanh()

    def forward(self, x, gt_color_source=None):
        """
        Args:
            x (Tensor): degraded input image
            gt_color_source (Tensor or None): GT image to extract color features from (used only during training)
        """
        # Use GT image to extract color feature if provided, else fallback to degraded input
        color_input = gt_color_source if gt_color_source is not None else x
        color_features = self.color_extractor(color_input)
        
        # Initial feature extraction from degraded image
        x = self.input(x)
        x = self.bn_input(x)
        x = self.hs_input(x)
        
        # Pass through ConvBlocks
        x = self.block1(x)
        x = self.block2(x)
        x = torch.cat([x, torch.zeros_like(x)[:, :16, :, :]], dim=1)  # Pad to 80 channels
        content_features = self.block3(x)  # Content features
        
        # Match resolution
        color_features = F.interpolate(color_features, size=content_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Apply CRM
        x = self.crm(content_features, color_features)
        
        # Final output
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
