# models/MAPs/multi_scale_prior.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    简单的 SE/Channel Attention：
    全局平均池化 → MLP → Sigmoid，得到通道权重
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.mlp(w)
        return x * w


class SpatialAttention(nn.Module):
    """
    空间注意力：
    通道维做 avg + max → concat → 7x7 conv → Sigmoid
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class MultiScalePriorAggregator(nn.Module):
    """
    我们自己的“多尺度先验聚合模块”（替代 MAPs 的 APA）：

    输入：
        x: 当前 decoder feature, B x C x H x W
        priors: 长度为3的列表 [S1, S2, S3]，分别是不同尺度的结构先验

    输出：
        aggregated_prior: B x out_channels x H x W
    """
    def __init__(
        self,
        feat_channels: int = 64,
        prior_channels=(64, 64, 64),
        out_channels: int = 64,
        reduction: int = 8,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        c1, c2, c3 = prior_channels

        # 先把三层先验都投影到与 x 相同的通道数
        self.proj1 = nn.Conv2d(c1, feat_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, feat_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, feat_channels, kernel_size=1)

        # 融合三个尺度：concat 后做 3x3 卷积
        self.fuse_conv = nn.Conv2d(feat_channels * 3, feat_channels, 3, padding=1)

        # 通道注意力 & 空间注意力
        self.channel_attn = ChannelAttention(feat_channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # 输出映射
        self.out_conv = nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, priors):
        """
        x:      B x C x H x W
        priors: [S1, S2, S3]，每个 S* 是 B x C_i x h x w
        """
        assert len(priors) == 3, "MultiScalePriorAggregator 目前假设正好三个尺度先验"

        B, C, H, W = x.shape

        p1, p2, p3 = priors

        # 统一到当前尺度 & 通道数
        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)

        p1 = self.proj1(p1)
        p2 = self.proj2(p2)
        p3 = self.proj3(p3)

        # 三尺度堆叠融合
        fused = torch.cat([p1, p2, p3], dim=1)      # B x (3C) x H x W
        fused = self.fuse_conv(fused)               # B x C x H x W

        # 通道注意力 + 空间注意力
        fused = self.channel_attn(fused)
        fused = self.spatial_attn(fused)

        # 输出为本层使用的结构 prior map
        out = self.out_conv(fused)
        return out
