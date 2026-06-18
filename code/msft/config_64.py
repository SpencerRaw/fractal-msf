# MSFT Configuration for 64×64 ImageNet experiments

# ─── Image dimensions ───
IMG_SIZE = 64
PATCH_SIZE = 4          # D₀: 4×4 high-res patch
MIDRES_SPAN = 16        # D₁: 16×16 region
MIDRES_DOWNSAMPLE = 4   # D₁: 16×16 ↓4 → 4×4 effective
FULL_DOWNSAMPLE = 16    # D₂: 64×64 ↓16 → 4×4 effective

# ─── FractalGen architecture for each MSFT level ───
# Each MSFT level receives a 4×4 "image" with the same token count (16)
# We use a small FractalGen: img_size_list=(4, 2, 1)
#   L0: 4×4 → 2×2 grid of 2×2 patches (4 tokens)
#   L1: 2×2 → 1×1 pixels (4 tokens)  
#   L2: pixel loss (3 tokens, RGB)
MSFT_LEVEL_IMG_SIZE_LIST = (4, 2, 1)
MSFT_LEVEL_EMBED_DIM_LIST = (384, 192, 64)    # ~1M params per level
MSFT_LEVEL_NUM_BLOCKS = (6, 3, 1)
MSFT_LEVEL_NUM_HEADS = (6, 3, 2)
MSFT_LEVEL_GENERATOR_TYPES = ("mar", "mar", "ar")  # MAR for top, AR for pixels

# ─── Data sampling ───
N_HIGHRES_PER_IMAGE = 5   # 每张图采 5 个 4×4 高清 patch
N_MIDRES_PER_IMAGE = 2    # 每张图采 2 个 16×16 中分辨 region

# ─── Coverage ratios (same as 256 setup) ───
# D₀: 4×4 / 64×64 = 16/4096 ≈ 0.4%
# D₁: 16×16 / 64×64 = 256/4096 ≈ 6.25%  
# D₂: 64×64 / 64×64 = 100%

# ─── FractalGen top-level model (for oracle/baseline) ───
# Standard FractalMAR-in64: img_size_list=(64, 4, 1)
# Used only for the "full data" oracle baseline
ORACLE_MODEL = "fractalmar_in64"

# ─── Training ───
BATCH_SIZE_PHASE1 = 64     # Phase 1 per-level batch size
BATCH_SIZE_PHASE2 = 32     # Phase 2 end-to-end batch size
LR_PHASE1 = 5e-5
LR_PHASE2 = 2e-5
EPOCHS_PHASE1 = 200        # Per-level pretraining epochs
EPOCHS_PHASE2 = 100        # End-to-end alignment epochs
