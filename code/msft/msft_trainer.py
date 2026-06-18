"""
MSFT Trainer: Multi-Scale Fragment Training for Fractal Generative Models.

支持三种训练模式：
  Phase 1: 逐层预训练 (Per-Level Pretraining)
  Phase 2: 端到端对齐 (End-to-End Alignment)
  DFS Generation: 深度优先生成完整高分辨率图像
"""

import os
import sys
import math
import time
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加 fractalgen 路径
FRACTALGEN_PATH = '/Users/apple/Desktop/loop/3跨尺度加速模拟/分型/fractalgen'
sys.path.insert(0, FRACTALGEN_PATH)

from models import fractalgen as fractalgen_models
from msft_dataset import MSFTDataset, MSFTPairedDataset, msft_collate_fn


class MSFTTrainer:
    """
    Multi-Scale Fragment Training wrapper around FractalGen.
    """

    def __init__(
        self,
        model_name: str = 'fractalmar_in64',
        img_size: int = 64,
        patch_size: int = 16,
        n_highres: int = 5,
        n_midres: int = 2,
        midres_span: int = 32,
        midres_downsample: int = 2,
        full_downsample: int = 4,
        device: str = 'cuda',
        output_dir: str = './output',
        **model_kwargs,
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_highres = n_highres
        self.n_midres = n_midres
        self.midres_span = midres_span
        self.midres_downsample = midres_downsample
        self.full_downsample = full_downsample
        self.device = device
        self.output_dir = output_dir
        
        # 创建模型
        self.model = fractalgen_models.__dict__[model_name](**model_kwargs)
        self.model.to(device)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[MSFT] Model: {model_name}, Params: {n_params/1e6:.1f}M")
        
        # ImageNet 归一化参数
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.global_step = 0

    def register_buffer(self, name, tensor):
        """Helper to move buffer to device."""
        setattr(self, name, tensor.to(self.device))

    # ════════════════════════════════════════════════════════════════
    # Phase 1: 逐层预训练
    # ════════════════════════════════════════════════════════════════

    def pretrain_level(self, data_root: str, level: int, epochs: int = 100,
                       batch_size: int = 64, lr: float = 5e-5,
                       log_writer: Optional[SummaryWriter] = None):
        """
        对 FractalGen 的某一层进行预训练。

        Args:
            data_root: ImageNet 数据路径
            level: 0=底层(高清局部), 1=中层(中分辨), 2=顶层(低清全图)
            epochs, batch_size, lr: 训练超参数
        """
        print(f"\n{'='*60}")
        print(f"[Phase 1] Pretraining level {level}")
        print(f"{'='*60}")

        # 准备数据：每层需要不同粒度的数据
        if level == 0:
            # 底层 g₀: 用高清 16×16 patch 训练
            # 需要先有上层模型来提供 condition
            # 简化版：用低清全图作为 condition
            dataset = MSFTDataset(
                root=data_root, img_size=self.img_size,
                patch_size=self.patch_size,
                n_highres=self.n_highres,
                train=True
            )
        elif level == 1:
            # 中层 g₁: 用中分辨数据训练
            dataset = MSFTDataset(
                root=data_root, img_size=self.img_size,
                patch_size=self.patch_size,
                n_midres=self.n_midres,
                midres_span=self.midres_span,
                midres_downsample=self.midres_downsample,
                train=True
            )
        else:
            # 顶层 g₂: 用低清全图训练
            dataset = MSFTDataset(
                root=data_root, img_size=self.img_size,
                patch_size=self.patch_size,
                n_highres=0, n_midres=0,
                train=True
            )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True,
                           collate_fn=msft_collate_fn)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for step, batch in enumerate(loader):
                optimizer.zero_grad()
                
                # 根据 level 选择对应的数据
                if level == 0:
                    # 底层：高清 patch
                    imgs = batch['highres'].to(self.device)
                    # 展平 batch 维度（n_highres × B）
                    B, N = imgs.shape[:2]
                    imgs = imgs.view(B * N, *imgs.shape[2:])
                    labels = batch['label'].repeat_interleave(N).to(self.device)
                elif level == 1:
                    imgs = batch['midres'].to(self.device)
                    B, N = imgs.shape[:2]
                    imgs = imgs.view(B * N, *imgs.shape[2:])
                    labels = batch['label'].repeat_interleave(N).to(self.device)
                else:
                    imgs = batch['full_low'].to(self.device)
                    labels = batch['label'].to(self.device)

                with torch.cuda.amp.autocast():
                    loss = self.model(imgs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()

                total_loss += loss.item()
                self.global_step += 1

            avg_loss = total_loss / len(loader)
            print(f"  Level {level} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
            if log_writer:
                log_writer.add_scalar(f'pretrain/level{level}_loss', avg_loss, epoch)

        # Save checkpoint
        ckpt_path = os.path.join(self.output_dir, f'pretrain_level{level}.pth')
        torch.save({'model': self.model.state_dict(), 'level': level}, ckpt_path)
        print(f"  [✓] Saved to {ckpt_path}")

    # ════════════════════════════════════════════════════════════════
    # Phase 2: 端到端对齐
    # ════════════════════════════════════════════════════════════════

    def align_end_to_end(self, data_root: str, epochs: int = 100,
                         batch_size: int = 32, lr: float = 2e-5,
                         log_writer: Optional[SummaryWriter] = None):
        """
        Phase 2: 用配对数据端到端训练所有层级。

        使用 MSFTPairedDataset（D₀ ⊂ D₁ ⊂ D₂ 空间对齐）。
        """
        print(f"\n{'='*60}")
        print(f"[Phase 2] End-to-End Alignment")
        print(f"{'='*60}")

        dataset = MSFTPairedDataset(
            root=data_root,
            img_size=self.img_size,
            patch_size=self.patch_size,
            n_levels=3,
            midres_span=self.midres_span,
            midres_downsample=self.midres_downsample,
            full_downsample=self.full_downsample,
            train=True,
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True,
                           collate_fn=msft_collate_fn)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for step, batch in enumerate(loader):
                optimizer.zero_grad()

                # BFS: 多层训练
                # 底层用高清 patch、中层用中分辨、顶层用低清全图
                losses = []

                # 顶层 loss: g₂ 预测低清全图
                imgs_top = batch['full_low'].to(self.device)
                labels = batch['label'].to(self.device)
                with torch.cuda.amp.autocast():
                    loss_top = self.model(imgs_top, labels)
                losses.append(loss_top)

                # 如果有中层数据，也加入
                if 'midres' in batch and batch['midres'].numel() > 0:
                    imgs_mid = batch['midres'].to(self.device)
                    with torch.cuda.amp.autocast():
                        loss_mid = self.model(imgs_mid, labels)
                    losses.append(loss_mid)

                # 如果有底层数据，也加入
                if 'highres' in batch and batch['highres'].numel() > 0:
                    imgs_high = batch['highres'].to(self.device)
                    with torch.cuda.amp.autocast():
                        loss_high = self.model(imgs_high, labels)
                    losses.append(loss_high)

                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()

                total_loss += loss.item()
                self.global_step += 1

            avg_loss = total_loss / len(loader)
            print(f"  Align | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
            if log_writer:
                log_writer.add_scalar('align/loss', avg_loss, epoch)

        ckpt_path = os.path.join(self.output_dir, 'aligned.pth')
        torch.save({'model': self.model.state_dict()}, ckpt_path)
        print(f"  [✓] Saved to {ckpt_path}")

    # ════════════════════════════════════════════════════════════════
    # DFS Generation
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(self, num_images: int = 16, class_labels=None,
                 cfg: float = 1.0, temperature: float = 1.0,
                 num_iter_list: str = '64,16') -> torch.Tensor:
        """
        DFS 生成完整图像。

        Args:
            num_images: 生成数量
            class_labels: 类别标签 (None = 随机)
            cfg: classifier-free guidance scale
            temperature: 采样温度
            num_iter_list: 每层 AR 迭代次数

        Returns:
            images: (num_images, 3, H, W) tensor, value range [0, 1]
        """
        self.model.eval()

        if class_labels is None:
            class_labels = torch.randint(0, 1000, (num_images,)).to(self.device)

        num_iter = [int(x) for x in num_iter_list.split(',')]

        # 使用原版 FractalGen 的 sample 方法
        images = self.model.sample(
            cond_list=class_labels,
            num_iter_list=num_iter,
            cfg=cfg,
            cfg_schedule='linear',
            temperature=temperature,
            filter_threshold=1e-4,
            fractal_level=0,
            visualize=False,
        )

        # Denormalize: [-1, 1] → [0, 1]
        # (FractalGen 内部有自己的 normalize 逻辑，这里按需调整)
        images = torch.clamp(images, 0.0, 1.0)
        
        return images

    def save_checkpoint(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'model_name': self.model_name,
            'img_size': self.img_size,
            'global_step': self.global_step,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.global_step = ckpt.get('global_step', 0)
        print(f"[MSFT] Loaded checkpoint from {path} (step {self.global_step})")
