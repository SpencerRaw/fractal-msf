# FractalMSF: Learning the Whole from Fragments

> Training Fractal Generative Models from Scattered Multi-Scale Patches
>
> **CV 前置工作** — 验证 "散落多尺度碎片训练分形生成模型" 可行性
> 为本人的 MD 跨尺度生成主论文提供方法论基础
>
> 项目路径: `~/Desktop/fractal-msf/`
> 创建日期: 2025-06-18
> 目标: 本周完成预实验核心验证

---

## 1. Project Overview

### 1.1 一句话

**仅用散落在不同分辨率的图像碎片（无完整全分辨率样本），训练 Fractal Generative Model 生成完整高分辨率图像。**

### 1.2 Motivation

在分子动力学等物理系统中，**完整的全原子超长尺度模拟极其昂贵或不可行**。研究者通常只有：

- 全原子 MD：极高保真，但时空覆盖极小（fs-ns，~10⁴ atoms）
- 粗粒化 MD：中等保真，中等覆盖（ns-μs，~10⁶ effective atoms）
- 连续介质/实验：低保真，全貌覆盖（μs-ms）

**问题**：能否用这些碎片数据训练一个跨尺度生成模型？

**策略**：先在图像域（ImageNet）上做 controlled experiment 验证可行性，再迁移到 MD。

### 1.3 核心类比

```
完整 256×256 高清图像  =  完整的 μs 级全原子轨迹  ← 我们没有

我们有的：
┌──────────────────────────────────────────────┐
│                                              │
│  低分辨全图 (256×256 → 16×16, 模糊)           │
│  = 连续介质 / 实验结构（全貌但极度粗糙）        │
│                                              │
│  ┌─────────────────┐                         │
│  │ 中分辨中等局部    │                        │
│  │ (64×64 → 16×16)  │  = CG MD              │
│  │ 覆盖~6%面积      │    (中等视野、中等保真)  │
│  │                  │                        │
│  │  ┌────┐          │                        │
│  │  │高清│          │  = 全原子 MD            │
│  │  │局部│          │    (极小视野、极高保真)  │
│  │  │16×16│         │                        │
│  │  └────┘          │                        │
│  └─────────────────┘                         │
│                                              │
└──────────────────────────────────────────────┘

三层 token 数相同 (= 16×16) → 天然对齐 FractalAR 序列长度
三层空间感受野不同 → 天然对应跨尺度 hierarchy
```

### 1.4 贡献命题

1. **新问题设定**：Multi-Scale Fragment Training (MSFT) — 从未被系统研究
2. **方法**：FractalMSF = FractalAR/MAR + 逐层对齐训练协议
3. **系统验证**：碎片覆盖率 / 层级数 / 训练协议的消融
4. **跨域 bridge**：证明该框架可迁移到分子模拟等物理域

---

## 2. Problem Formulation

### 2.1 Notation

| 符号 | 含义 |
|---|---|
| x ∈ ℝ^{H×W×3} | 完整全分辨率图像（H=W=256） |
| D₀ | 高清局部 patch 集合：{x₀^(i) ∈ ℝ^{16×16×3}, 原始分辨率} |
| D₁ | 中分辨局部 patch 集合：{x₁^(i) ∈ ℝ^{64×64×3} ↓4, 等效 16×16} |
| D₂ | 低清全图集合：{x₂^(i) ∈ ℝ^{256×256×3} ↓16, 等效 16×16} |
| N₀, N₁, N₂ | 各类数据样本数 |

### 2.2 Key Properties of MSFT Data

```
Property 1: 信息量守恒
  |D₀|_tokens = |D₁|_tokens = |D₂|_tokens = 16×16 = 256 tokens
  但空间覆盖面积: 16² ≪ 64² ≪ 256²

Property 2: 碎片间无显式空间对齐
  D₀ 中的 patch 和 D₁ 中的 region 和 D₂ 中的全图
  不需要来自同一张原图的同一位置
  （可利用 MAR 的 permutation invariance）

Property 3: 覆盖不完全
  某些空间区域在任何分辨率下都没有数据
  → 模型必须在不同尺度间"脑补"缺失区域
```

### 2.3 Training Objective

学 p_θ(x_full_highres)，仅用 D₀ ∪ D₁ ∪ D₂。

---

## 3. Method: FractalMSF

### 3.1 Base Architecture: FractalMAR

基于 Li et al. (2025) 的 Fractal Generative Models：

```
Level 2 (顶层): AR/BERT → 生成 16×16 低清全图 context
  ↓
Level 1 (中层): AR/BERT → 在低清 context 下生成 64×64 中分辨局部
  ↓
Level 0 (底层): AR/BERT → 在中分辨 context 下生成 16×16 高清局部
  ↓
Loss: CE on discrete pixel values (0-255, 256-way classification)
```

每一层的 AR 模型序列长度 = 256 (16×16)，架构完全相同（不同 scale embedding）。

### 3.2 Training Protocol

```
═══════════════════════════════════════════
PHASE 1: 逐层预训练 (Per-Level Pretraining)
═══════════════════════════════════════════

Train g₂:  D₂ (低清全图 256×256↓16) → CE loss
  - 目标：学 p(低清全图的 pixel 序列)
  - 输出：context vectors for g₁

Train g₁:  D₁ (中分辨 64×64↓4) → CE loss  
  - 目标：学 p(中分辨 patch | 低清全图 context)
  - 用 g₂ 的 frozen output 作为条件
  - 输出：context vectors for g₀

Train g₀:  D₀ (高清 16×16) → CE loss
  - 目标：学 p(高清 patch | 中分辨 context)
  - 用 g₁ 的 frozen output 作为条件

═══════════════════════════════════════════
PHASE 2: 端到端对齐 (End-to-End Alignment)
═══════════════════════════════════════════

构造配对数据：
  从同一张原图中取出：
    1 个 256×256 低清全图 (D₂)
    k 个 64×64 中分辨区域 (D₁) 
    k×4 个 16×16 高清 patch (D₀)
  确保 D₀ ⊂ D₁ ⊂ D₂ 空间包含关系

BFS 展开 3-level 架构 → CE loss 反传所有层级
可选：中间层不加显式 loss（仅底层梯度反传）

═══════════════════════════════════════════
PHASE 3: DFS 生成 (Depth-First Generation)
═══════════════════════════════════════════

g₂ 生成 256×256 低清全图 (16×16 grid of context vectors)
  ↓
for each 64×64 region:
    g₁ 生成该 region 的中分辨表示 (4×4 grid)
  ↓
for each 16×16 patch:
    g₀ 生成该 patch 的高清像素

→ 完整 256×256 高清图像
```

### 3.3 Baseline Methods

| Method | 使用的数据 | 描述 |
|---|---|---|
| **Oracle (Upper Bound)** | 完整 256×256 高清 | 原版 FractalMAR |
| **SuperRes Only** | D₂ only (低清全图) | 传统超分：低清 → 高清 |
| **Outpaint Only** | D₀ only (高清局部) | 传统外推：局部 → 全图 |
| **MSFT 2-level** | D₀ + D₂ (无中层) | 消融：中层是否必要 |
| **MSFT 3-level (Ours)** | D₀ + D₁ + D₂ | 完整 FractalMSF |
| **MSFT Random Init** | D₀ + D₁ + D₂ | 无逐层预训练，直接端到端 |

---

## 4. Experimental Plan

### 4.1 Datasets

- **Primary**: ImageNet 64×64 (unconditional) — 快速迭代
- **Main**: ImageNet 256×256 (class-conditional) — 最终评估

### 4.2 Data Construction

从每张原图 (256×256) 构造 MSFT 数据：

```
D₀ (高清局部): 
  随机采样 m 个不重叠的 16×16 patch (m = 5 default)
  → N₀ = N_images × 5

D₁ (中分辨局部):
  随机采样 n 个 64×64 region，下采样 4× 到 16×16 (n = 2 default)
  → N₁ = N_images × 2

D₂ (低清全图):
  整张 256×256 下采样 16× 到 16×16
  → N₂ = N_images × 1
```

### 4.3 Experiment Matrix

#### Exp 1: Feasibility Check (本周)
**Q**: 2-level MSFT 能否生成合理图像？

```
Setting: ImageNet 64×64
Data: D₀ (16×16 高清 patch) + D₂ (64×64 ↓4 低清全图)
Model: FractalMAR 2-level
Epochs: 100 (逐层) + 100 (端到端)
Metric: FID vs Oracle vs SuperRes Only
```

#### Exp 2: 3-level vs 2-level
**Q**: 中层 D₁ 是否带来额外收益？

```
Data: 3-level (D₀ + D₁ + D₂) vs 2-level (D₀ + D₂)
Metric: FID, IS, LPIPS
```

#### Exp 3: Fragment Coverage
**Q**: 多少高清 patch 才够？

```
m ∈ {1, 5, 10, 25, 50} patches per image
→ FID vs coverage curve
→ 找临界点
```

#### Exp 4: Training Protocol Ablation
**Q**: 逐层预训练 vs 端到端 vs 混合？

```
Protocol A: 逐层 only (Phase 1)
Protocol B: 端到端 only (no Phase 1)  
Protocol C: 逐层 + 端到端 (Phase 1 + 2)
→ FID, training stability
```

#### Exp 5: Spatial Alignment
**Q**: 碎片间是否需要显式空间对齐？

```
Aligned: D₀ ⊂ D₁ ⊂ D₂（来自原图同一区域）
Unaligned: D₀, D₁, D₂ 来自随机位置
→ FID 差异
```

#### Exp 6: Scaling Study
**Q**: 模型规模 (S/M/L) → 生成质量？

```
FractalMAR-B (186M) / L (438M) / H (848M)
→ FID vs params
```

### 4.4 Evaluation Metrics

| Metric | 测什么 | 目标 |
|---|---|---|
| FID↓ | 生成质量 + 多样性 | < 10 (2-level), < 7 (3-level) |
| IS↑ | 清晰度 + 多样性 | > 100 |
| Precision↑ | 保真度 (fidelity) | > 0.7 |
| Recall↑ | 多样性 (diversity) | > 0.3 |
| LPIPS↓ | 感知相似度 (vs GT) | < 0.3 |
| PSNR/SSIM↑ | 像素级重建 (vs GT，仅配对评估) | > 25 dB |

### 4.5 Success Criteria

| 阶段 | Criterion |
|---|---|
| **本周预实验** | 2-level MSFT FID < 15，显著优于 SuperRes Only |
| **完整实验** | 3-level MSFT FID < 8，接近 Oracle 的 50% 以内 |
| **可投稿** | 完整消融 + scaling study + 定性分析 |

---

## 5. Code Architecture

### 5.1 代码基础

基于 https://github.com/LTH14/fractalgen (MIT License)

### 5.2 需要改动/新增的文件

```
fractal-msf/
├── code/
│   ├── fractalgen/          # clone 的原版仓库
│   ├── msft/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── msft_dataset.py      # MSFT data loader (核心改动)
│   │   │   ├── sampling.py          # 多尺度碎片采样逻辑
│   │   │   └── transforms.py        # 下采样 / 数据增强
│   │   ├── models/
│   │   │   ├── fractal_msf.py       # FractalMSF 模型 (多级训练包装)
│   │   │   └── level_trainer.py     # 单层训练器
│   │   ├── training/
│   │   │   ├── phase1_pretrain.py   # Phase 1: 逐层预训练
│   │   │   ├── phase2_align.py      # Phase 2: 端到端对齐
│   │   │   └── config.py            # 训练配置
│   │   ├── generation/
│   │   │   └── dfs_generator.py     # DFS 生成完整图像
│   │   └── eval/
│   │       ├── fid_eval.py          # FID 评估
│   │       └── metrics.py           # LPIPS, PSNR, SSIM
│   └── scripts/
│       ├── run_pretrain.sh          # Phase 1 启动脚本
│       ├── run_align.sh             # Phase 2 启动脚本
│       └── run_eval.sh              # 评估脚本
├── data/
│   ├── imagenet64/                  # ImageNet 64×64
│   └── imagenet256/                 # ImageNet 256×256
├── experiments/
│   ├── exp1_feasibility/            # 本周预实验
│   ├── exp2_3level/                 # 3-level 验证
│   └── exp3_coverage/               # 覆盖率消融
├── notes/
│   ├── daily_log.md                 # 实验日志
│   └── issues.md                    # 遇到的问题
└── figures/
    └── teaser/                      # 论文配图草稿
```

### 5.3 Key Code Snippets

#### MSFT Data Loader (核心)

```python
class MSFTDataset(Dataset):
    """Multi-Scale Fragment Training dataset."""
    
    def __init__(self, image_dir, 
                 patch_size=16,        # D₀ 的 patch 大小
                 medium_size=64,       # D₁ 的空间跨度
                 full_size=256,        # D₂ 的空间跨度
                 n_patches=5,          # 每张图采多少个高清 patch
                 n_medium=2,           # 每张图采多少个中分辨 region
                 medium_downsample=4,  # D₁ 的下采样倍数
                 full_downsample=16):  # D₂ 的下采样倍数
        ...
    
    def __getitem__(self, idx):
        img = load_image(idx)  # 256×256×3
        
        # D₂: 低清全图
        full_low = F.interpolate(img, scale_factor=1/self.full_downsample)
        # → 16×16×3
        
        # D₁: 中分辨局部
        medium_regions = []
        for _ in range(self.n_medium):
            region = random_crop(img, self.medium_size)  # 64×64
            region_low = F.interpolate(region, scale_factor=1/self.medium_downsample)
            # → 16×16×3
            medium_regions.append(region_low)
        
        # D₀: 高清局部
        patches = []
        for _ in range(self.n_patches):
            patch = random_crop(img, self.patch_size)  # 16×16
            patches.append(patch)  # 保持原始分辨率
        
        return {
            'full_low': full_low,        # (16, 16, 3)
            'medium_regions': medium_regions,  # [(16, 16, 3)]
            'patches': patches,          # [(16, 16, 3)]
        }
```

#### DFS 生成

```python
def generate_full_image(model, n_levels=3):
    """
    Depth-first generation of full high-res image.
    """
    # Level 2: 生成低清全图 (16×16 grid)
    context_l2 = model.g2.generate()  # (16, 16, context_dim)
    
    # 初始化输出 canvas (256×256)
    output = torch.zeros(256, 256, 3)
    
    # Level 1: 在每个 64×64 region 内细化
    for i in range(4):       # 256/64 = 4
        for j in range(4):
            # 取出该 region 对应的 4×4 low-res context
            region_ctx = context_l2[i*4:(i+1)*4, j*4:(j+1)*4]
            
            # Level 0: 在 16×16 patch 内生成高清像素
            for di in range(4):   # 64/16 = 4
                for dj in range(4):
                    patch_ctx = model.g1.generate(cond=region_ctx)
                    pixels = model.g0.generate(cond=patch_ctx)
                    output[i*64+di*16:(i*64+(di+1)*16),
                           j*64+dj*16:(j*64+(dj+1)*16)] = pixels
    
    return output
```

---

## 6. Timeline

### Week 1 (本周, Jun 16-22): 预实验

| Day | Task | Checkpoint |
|---|---|---|
| Mon (today) | 项目搭建 + 代码 clone + 数据准备 | 跑通原版 FractalMAR 64×64 |
| Tue | 改造 data loader → MSFT 2-level | 能正确采样碎片 |
| Wed | Phase 1: 逐层预训练 g₂ + g₀ | 每层 loss 下降 |
| Thu | Phase 2: 端到端对齐 | 联合 loss 下降 |
| Fri | DFS 生成 + FID 评估 | 拿到第一个 FID 数字 |
| Sat-Sun | 3-level 扩展 + 快速消融 | 基础结论 |

### Week 2 (Jun 23-29): 系统实验

- 3-level 完整训练
- 覆盖率消融 (m=1/5/10/25/50)
- 训练协议消融
- Baseline (SuperRes Only, Outpaint Only)

### Week 3 (Jun 30 - Jul 6): 分析与写作

- Scaling study (B/L/H)
- 定性分析（可视化生成过程）
- Draft paper outline
- Figure preparation

### Week 4 (Jul 7-13): 论文撰写

- Full draft
- Related work
- Appendix

---

## 7. References

- **Fractal Generative Models** — Li et al., arXiv:2502.17437v2 (2025)
- **MAR** — Li et al., "Autoregressive Image Generation without Vector Quantization" (2024)
- **VAR** — Tian et al., "Visual Autoregressive Modeling" (2024)
- **MegaByte** — Yu et al., "MegaByte: Predicting Million-byte Sequences" (2023)
- FractalNet — Larsson et al. (2016)
- ImageNet — Deng et al. (2009)

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 2-level FID > 20 (差距过大) | Medium | 增加高清 patch 数量 / 降低难度到 32×32 |
| Phase 1 逐层训练不稳定 | Medium | 降低 lr / 增加 warmup |
| 计算资源不足 | Low | 先在 64×64 跑通，确认可行后再申请 256×256 资源 |
| 代码改动超过预期 | Medium | 最小改动原则：只改 data loader + training loop |
| DFS 生成质量差 | Medium | 加入 patch boundary smoothing (如 Kaiming 的 neighbor context) |
