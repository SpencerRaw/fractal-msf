# FractalMSF: Learning the Whole from Fragments

**Multi-Scale Fragment Training for Fractal Generative Models.**

[![arXiv](https://img.shields.io/badge/arXiv-2502.17437-b31b1b.svg)](https://arxiv.org/abs/2502.17437)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Can we train a generative model to produce **complete high-resolution images** using only **scattered multi-scale fragments** — patches at different resolutions that don't cover the full image?

This is the MSFT (Multi-Scale Fragment Training) problem setting. It mirrors real-world constraints in molecular dynamics, medical imaging, and remote sensing where full high-res data is unavailable.

## Approach

We build on [Fractal Generative Models](https://github.com/LTH14/fractalgen) (Li et al., 2025) and introduce a **per-level pretraining + end-to-end alignment** protocol that trains fractal generators from heterogeneous scale-specific data.

```
Full 256×256 image  ←  we DON'T have this

What we have:
┌──────────────────────────────┐
│  Low-res full (16×16 eff.)    │  ← D₂: continuum / experiment
│  ┌──────────────┐             │
│  │ Mid-res local│             │  ← D₁: coarse-grained MD
│  │ (16×16 eff.) │             │
│  │  ┌────┐      │             │
│  │  │High│      │             │  ← D₀: all-atom MD
│  │  │res │      │             │
│  │  └────┘      │             │
│  └──────────────┘             │
└──────────────────────────────┘
```

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SpencerRaw/fractal-msf/blob/main/code/FractalMSF_Training.ipynb)

```bash
git clone https://github.com/SpencerRaw/fractal-msf.git
cd fractal-msf

# Install dependencies
pip install torch torchvision timm torch_fidelity tensorboard

# Run on Colab (recommended) — open code/FractalMSF_Training.ipynb
```

## Project Structure

```
fractal-msf/
├── PLAN.md                           # Full project plan & timeline
├── code/
│   ├── FractalMSF_Training.ipynb     # Colab training notebook
│   └── msft/
│       ├── msft_dataset.py           # Multi-scale fragment dataset
│       └── msft_trainer.py           # Training wrapper (pretrain + align)
├── experiments/                      # Experiment logs
└── figures/                          # Paper figures
```

## Based On

- [Fractal Generative Models](https://github.com/LTH14/fractalgen) — Li, Sun, Fan, He (MIT CSAIL / Google DeepMind, 2025)

## License

MIT
