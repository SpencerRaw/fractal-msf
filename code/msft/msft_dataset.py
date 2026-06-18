"""
MSFT Dataset: Multi-Scale Fragment Training.

从完整图像中采样散落在不同尺度的碎片：
  D₀: 高清小局部 (16×16, 原始分辨率)
  D₁: 中分辨中等局部 (64×64 ↓4 → 16×16 effective)
  D₂: 低清全图 (256×256 ↓16 → 16×16 effective)

所有尺度的 token 数相同 = patch_size × patch_size。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image


class MSFTDataset(Dataset):
    """
    Multi-Scale Fragment Training dataset wrapper.
    
    对每张原图，采样多尺度碎片。返回一个 dict，包含各尺度的 patch。
    """

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        patch_size: int = 16,
        n_highres: int = 5,          # 每张图采多少个高清 patch (D₀)
        n_midres: int = 2,            # 每张图采多少个中分辨 region (D₁)
        midres_span: int = 64,        # D₁ 的空间跨度 (64×64 区域)
        midres_downsample: int = 4,   # D₁ 下采样倍数 (64/4=16 effective)
        full_downsample: int = 16,    # D₂ 下采样倍数 (256/16=16 effective)
        train: bool = True,
        transform_base=None,          # 基础 transform (如随机水平翻转)
        normalize=True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_highres = n_highres
        self.n_midres = n_midres
        self.midres_span = midres_span
        self.midres_downsample = midres_downsample
        self.full_downsample = full_downsample
        self.train = train
        self.normalize = normalize
        
        # ImageNet normalization
        if normalize:
            self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 加载原始数据集
        split = 'train' if train else 'val'
        dataset_path = f"{root}/{split}" if not root.endswith(split) else root
        
        if transform_base is None:
            transform_base = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size) if not train else transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])
        
        self.dataset = ImageFolder(dataset_path, transform=transform_base)
        
        # 验证参数
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        assert img_size % midres_span == 0, f"img_size ({img_size}) must be divisible by midres_span ({midres_span})"
        assert midres_span % midres_downsample == 0, "midres_span must be divisible by midres_downsample"
        assert midres_span // midres_downsample == patch_size, \
            f"midres effective size ({midres_span//midres_downsample}) must equal patch_size ({patch_size})"
        assert img_size // full_downsample == patch_size, \
            f"full effective size ({img_size//full_downsample}) must equal patch_size ({patch_size})"

    def __len__(self):
        return len(self.dataset)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        if self.normalize:
            return (img - self.norm_mean) / self.norm_std
        return img

    def _denormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet normalization."""
        if self.normalize:
            return img * self.norm_std + self.norm_mean
        return img

    def _random_crop(self, img: torch.Tensor, size: int) -> torch.Tensor:
        """随机裁切一个 size×size 的区域。"""
        _, h, w = img.shape
        if h == size and w == size:
            return img
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return img[:, top:top+size, left:left+size]

    def _downsample(self, img: torch.Tensor, factor: int) -> torch.Tensor:
        """下采样 factor 倍 (area interpolation, 等价于平均池化)."""
        _, h, w = img.shape
        img_4d = img.unsqueeze(0)  # (1, C, H, W)
        down = F.interpolate(img_4d, size=(h//factor, w//factor), 
                            mode='area').squeeze(0)
        return down

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # img: (3, H, W), normalized
        
        result = {'label': label}
        
        # ── D₂: 低清全图 ──
        # 整张图 ↓16 → patch_size × patch_size
        full_low = self._downsample(img, self.full_downsample)
        result['full_low'] = full_low  # (3, patch_size, patch_size)
        
        # ── D₁: 中分辨中等局部 ──
        # 裁切 midres_span×midres_span 区域 → ↓midres_downsample → patch_size × patch_size
        midres_list = []
        for _ in range(self.n_midres):
            region = self._random_crop(img, self.midres_span)
            region_low = self._downsample(region, self.midres_downsample)
            midres_list.append(region_low)  # (3, patch_size, patch_size)
        result['midres'] = torch.stack(midres_list) if midres_list else torch.empty(0)
        
        # ── D₀: 高清小局部 ──
        # 裁切 patch_size × patch_size 区域（原始分辨率，不下采样）
        highres_list = []
        for _ in range(self.n_highres):
            patch = self._random_crop(img, self.patch_size)
            highres_list.append(patch)  # (3, patch_size, patch_size)
        result['highres'] = torch.stack(highres_list) if highres_list else torch.empty(0)
        
        return result


class MSFTPairedDataset(Dataset):
    """
    Paired MSFT dataset for Phase 2 (end-to-end alignment).
    
    从同一张原图中采样空间对齐的多尺度碎片：
      - D₀ ⊂ D₁ ⊂ D₂（空间包含关系）
    
    用于 BFS 端到端训练时保证层间一致性。
    """

    def __init__(
        self,
        root: str,
        img_size: int = 256,
        patch_size: int = 16,
        n_levels: int = 3,
        midres_span: int = 64,
        midres_downsample: int = 4,
        full_downsample: int = 16,
        train: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.midres_span = midres_span
        self.midres_downsample = midres_downsample
        self.full_downsample = full_downsample
        
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        split = 'train' if train else 'val'
        dataset_path = f"{root}/{split}"
        
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        
        self.dataset = ImageFolder(dataset_path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def _normalize(self, img):
        return (img - self.norm_mean) / self.norm_std

    def _downsample(self, img, factor):
        return F.interpolate(img.unsqueeze(0), 
                            scale_factor=1.0/factor, 
                            mode='area').squeeze(0)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self._normalize(img)
        
        result = {'label': label}
        
        # 随机选一个 midres_span×midres_span 的锚定区域
        grid_size = self.img_size // self.midres_span  # 256/64 = 4
        gi = np.random.randint(0, grid_size)
        gj = np.random.randint(0, grid_size)
        
        # D₂: 低清全图
        full_low = self._downsample(img, self.full_downsample)
        result['full_low'] = full_low
        
        # D₁: 中分辨区域（对齐）
        y1, y2 = gi * self.midres_span, (gi + 1) * self.midres_span
        x1, x2 = gj * self.midres_span, (gj + 1) * self.midres_span
        region = img[:, y1:y2, x1:x2]
        midres_low = self._downsample(region, self.midres_downsample)
        result['midres'] = midres_low  # (3, 16, 16)
        
        # D₀: 高清局部（在 anchor region 内随机选一个 patch）
        if self.n_levels >= 3:
            sub_grid = self.midres_span // self.patch_size  # 64/16 = 4
            si = np.random.randint(0, sub_grid)
            sj = np.random.randint(0, sub_grid)
            py1, py2 = si * self.patch_size, (si + 1) * self.patch_size
            px1, px2 = sj * self.patch_size, (sj + 1) * self.patch_size
            patch = region[:, py1:py2, px1:px2]
            result['highres'] = patch  # (3, 16, 16)
        
        return result


def msft_collate_fn(batch):
    """
    Collate function for MSFT dataset.
    将 list of dicts 合并为 batched dict。
    """
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            if values[0].dim() == 3:
                # (C, H, W) → (B, C, H, W)
                result[key] = torch.stack(values)
            elif values[0].dim() == 4:
                # Already (N, C, H, W) → flatten batch dim
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values)
    return result
