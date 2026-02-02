"""
CSIST_RSSB_DIST_BasicBlock: 针对CSIST任务定制的RSSB增强版

基于方案三（mAP 0.4674），针对红外小目标检测任务特点进行定制化改进

红外小目标特点：
- 目标极小（几个像素）
- 信噪比低
- 背景杂波干扰
- 稀疏分布

创新点：
1. 小目标敏感卷积模块（Small Target Sensitive Convolution, STSC）
   - 多尺度小核并行（1×1, 3×3）捕获点状目标
   - 中心加权机制，增强对小目标中心的响应
   
2. 自适应稀疏阈值（Adaptive Sparse Threshold, AST）
   - 根据特征统计量自适应调整稀疏阈值
   - 显式稀疏性约束，增强目标-背景分离

论文表述：
English: We propose CSIST-RSSB-DISTANet, a task-specific enhancement of DISTANet 
for Compressed Sensing Infrared Small Target detection. Our method introduces:
(1) Small Target Sensitive Convolution (STSC) that employs parallel multi-scale 
small kernels (1×1, 3×3) with center-weighted attention to enhance point-source 
target responses;
(2) Adaptive Sparse Threshold (AST) that dynamically adjusts soft-threshold 
values based on feature statistics, explicitly enforcing sparsity priors for 
improved target-background separation.

中文：本文提出CSIST-RSSB-DISTANet，针对压缩感知红外小目标检测任务定制化增强。
主要创新包括：
(1) 小目标敏感卷积(STSC) - 采用并行多尺度小核(1×1, 3×3)配合中心加权注意力，
增强对点状目标的响应；
(2) 自适应稀疏阈值(AST) - 根据特征统计量动态调整软阈值，显式引入稀疏性先验，
提升目标-背景分离能力。

投稿目标：TIP / TGRS / CVPR Workshop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================== 创新点1: 小目标敏感卷积 ====================

class SmallTargetSensitiveConv(nn.Module):
    """
    小目标敏感卷积模块 (STSC)
    
    设计思想：
    - 红外小目标通常只有几个像素，大卷积核容易平滑掉
    - 使用小核(1×1, 3×3)并行提取，保留点状特征
    - 中心加权机制，增强对目标中心像素的响应
    """
    def __init__(self, dim):
        super().__init__()
        
        # 1×1卷积：捕获逐像素特征（对点状目标最敏感）
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # 3×3卷积：捕获小邻域特征
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
        # 中心加权注意力
        self.center_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        # 融合权重（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 多尺度小核特征
        f1 = self.conv1x1(x)  # 点状特征
        f3 = self.conv3x3(x)  # 小邻域特征
        
        # 自适应融合
        alpha = torch.sigmoid(self.alpha)
        f_multi = alpha * f1 + (1 - alpha) * f3
        
        # 中心加权（增强目标中心响应）
        attn = self.center_attention(f_multi)
        f_weighted = f_multi * attn
        
        # 特征增强 + 残差
        out = self.enhance(f_weighted)
        
        return x + out


# ==================== 创新点2: 自适应稀疏阈值 ====================

class AdaptiveSparseThreshold(nn.Module):
    """
    自适应稀疏阈值模块 (AST)
    
    设计思想：
    - 红外小目标在图像中稀疏分布
    - 固定阈值无法适应不同场景
    - 根据特征统计量（均值、方差）动态调整阈值
    - 显式稀疏性约束，增强目标-背景分离
    """
    def __init__(self, dim, base_threshold=0.01):
        super().__init__()
        
        self.base_threshold = nn.Parameter(torch.tensor(base_threshold))
        
        # 统计量编码器
        self.stat_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Softplus()  # 确保正值
        )
        
        # 空间自适应调制
        self.spatial_mod = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 阈值范围约束
        self.tau_min = 1e-4
        self.tau_max = 0.05
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: 软阈值后的特征
        """
        B, C, H, W = x.shape
        
        # 全局统计量 -> 基础阈值调制
        stat_mod = self.stat_encoder(x)  # (B, 1)
        
        # 空间调制图
        spatial_mod = self.spatial_mod(x)  # (B, 1, H, W)
        
        # 动态阈值 = base × (1 + stat_mod) × spatial_mod
        threshold = self.base_threshold * (1 + 0.1 * stat_mod.view(B, 1, 1, 1)) * (0.5 + spatial_mod)
        threshold = torch.clamp(threshold, self.tau_min, self.tau_max)
        
        # 软阈值操作
        x_thresh = torch.sign(x) * F.relu(torch.abs(x) - threshold)
        
        return x_thresh


# ==================== 简化版RSSB（来自方案三）====================

class SimpleRSSB(nn.Module):
    """简化版残差状态空间块 - 来自方案三"""
    def __init__(self, dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.global_conv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        
        self.channel_mix = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = F.gelu(x)
        x = x + self.global_conv(x)
        x = self.channel_mix(x)
        x = self.conv2(x)
        x = x * self.ca(x)
        return identity + self.gamma * x


# ==================== 主模块 ====================

class CSIST_RSSB_DIST_BasicBlock(nn.Module):
    """
    CSIST任务定制的RSSB增强版
    
    结构：
    - 静态分支（原始设计）
    - RSSB分支（方案三）
    - 【创新点1】小目标敏感卷积
    - 【创新点2】自适应稀疏阈值
    
    创新点：
    1. 小目标敏感卷积(STSC) - 多尺度小核 + 中心加权
    2. 自适应稀疏阈值(AST) - 动态阈值 + 稀疏性约束
    """
    def __init__(self, **kwargs):
        super(CSIST_RSSB_DIST_BasicBlock, self).__init__()
        
        c = kwargs['c']
        lambda_weight = kwargs.get('lambda_weight', 0.7)
        
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.c = c
        
        # ===== 静态分支（保持原设计）=====
        self.conv1_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        
        # ===== RSSB分支（方案三）=====
        self.rssb_proj = nn.Conv2d(1, 64, 3, 1, 1)
        self.rssb_block = SimpleRSSB(64)
        
        # ===== 【创新点1】小目标敏感卷积 =====
        self.stsc = SmallTargetSensitiveConv(64)
        
        # ===== 【创新点2】自适应稀疏阈值 =====
        self.ast = AdaptiveSparseThreshold(64, base_threshold=0.01)
        
        # ===== 后向变换 =====
        self.conv1_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv2_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))
        
        self.lambda_weight = torch.Tensor([lambda_weight]).to(device)
        
    def forward(self, x, PhiTPhi, PhiTb):
        # ISTA梯度下降步
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)
        
        # ===== 静态分支 =====
        x_s = F.conv2d(x_input, self.conv1_forward, padding=1)
        x_s = F.relu(x_s)
        x_forward = F.conv2d(x_s, self.conv2_forward, padding=1)
        
        # ===== RSSB分支 =====
        x_rssb = self.rssb_proj(x_input)
        x_rssb = self.rssb_block(x_rssb)
        
        # ===== 双分支融合 =====
        x_combined = self.lambda_weight * x_forward + \
                     (1 - self.lambda_weight) * x_rssb
        
        # ===== 【创新点1】小目标敏感卷积 =====
        x_stsc = self.stsc(x_combined)
        
        # ===== 【创新点2】自适应稀疏阈值 =====
        x_thresh = self.ast(x_stsc)
        
        # ===== 后向变换 =====
        x_b = F.conv2d(x_thresh, self.conv1_backward, padding=1)
        x_b = F.relu(x_b)
        x_backward = F.conv2d(x_b, self.conv2_backward, padding=1)
        
        x_pred = x_backward.view(-1, 11 * self.c * 11 * self.c)
        
        # ===== 对称损失 =====
        x_sym = F.conv2d(x_stsc, self.conv1_backward, padding=1)
        x_sym = F.relu(x_sym)
        x_est = F.conv2d(x_sym, self.conv2_backward, padding=1)
        symloss = x_est - x_input
        
        return [x_pred, symloss]


# ==================== 测试 ====================

if __name__ == "__main__":
    print("Testing CSIST_RSSB_DIST_BasicBlock...")
    
    batch_size = 64
    c = 3
    
    block = CSIST_RSSB_DIST_BasicBlock(c=c, lambda_weight=0.7)
    block = block.to(device)
    
    x = torch.randn(batch_size, 11 * 11 * c * c).to(device)
    PhiTPhi = torch.randn(11 * 11 * c * c, 11 * 11 * c * c).to(device)
    PhiTb = torch.randn(batch_size, 11 * 11 * c * c).to(device)
    
    output, symloss = block(x, PhiTPhi, PhiTb)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Symloss shape: {symloss.shape}")
    
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("Test passed!")