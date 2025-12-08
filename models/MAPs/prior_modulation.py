# models/MAPs/prior_modulation.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorAffineModulation(nn.Module):
    """
    先验引导的仿 SPADE 仿射调制：
    - 对 x 做 InstanceNorm
    - 用 prior 生成 gamma / beta
    - 输出: x_norm * (1 + gamma) + beta
    """
    def __init__(self, feat_channels: int, prior_channels: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feat_channels, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden, feat_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden, feat_channels, kernel_size=3, padding=1)

    def forward(self, x, prior):
        # x: B x C x H x W
        # prior: B x C_p x h x w
        prior = F.interpolate(prior, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x_norm = self.norm(x)

        h = self.mlp_shared(prior)
        gamma = self.mlp_gamma(h)
        beta = self.mlp_beta(h)

        out = x_norm * (1.0 + gamma) + beta
        return out


class GatedConv2d(nn.Module):
    """
    带门控的卷积：输出一半作为特征，一半作为 gate
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv(x)
        feat, gate = torch.split(x, self.out_channels, dim=1)
        return feat * torch.sigmoid(gate)


class StructureAwareModulation(nn.Module):
    """
    我们自己的“结构先验引导调制模块”（替代 BAM）：

    输入：
        x:      B x C x H x W  （decoder 当前特征）
        prior:  B x C_p x h x w（由 MultiScalePriorAggregator 输出的结构先验图）

    输出：
        out:    B x out_channels x H x W
    """
    def __init__(
        self,
        feat_channels: int = 64,
        prior_channels: int = 64,
        out_channels: int = 64,
        use_concat: bool = True,
    ):
        super().__init__()
        self.use_concat = use_concat
        self.out_channels = out_channels

        # 预处理
        self.conv_x = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.conv_p = nn.Conv2d(prior_channels, prior_channels, kernel_size=3, padding=1)

        # 1) 仿 SPADE 仿射调制分支
        self.affine_mod = PriorAffineModulation(
            feat_channels=feat_channels,
            prior_channels=prior_channels,
            hidden=128
        )

        # 2) 门控卷积分支（结构与纹理的 gated 融合）
        self.gated_conv = nn.Sequential(
            nn.Conv2d(feat_channels + prior_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            GatedConv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        )

        # 融合两个分支
        self.reduce = nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1, bias=False)

        # 最终 refinement
        self.out_conv = nn.Sequential(
            nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, prior):
        """
        x:     B x C x H x W
        prior: B x C_p x h x w
        """
        prior = F.interpolate(prior, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x_ = self.conv_x(x)
        p_ = self.conv_p(prior)

        # 分支1：仿射调制
        affine_out = self.affine_mod(x_, p_)

        # 分支2：门控卷积
        gated_in = torch.cat([x_, p_], dim=1)
        gated_out = self.gated_conv(gated_in)

        # 融合两支
        fused = self.reduce(torch.cat([affine_out, gated_out], dim=1))

        out = self.out_conv(fused)
        return out
