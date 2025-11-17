# Differentiable-Patterning Tutorials

Welcome! This guide will help you learn how to use the Differentiable-Patterning library for research into self-organizing systems and emergent pattern formation.

## What is This Library?

This library enables you to:
- Train **Neural Cellular Automata (NCA)** to generate specific patterns
- Simulate and learn **Partial Differential Equations (PDEs)** that produce pattern formation
- Use gradient-based optimization to discover pattern-forming systems
- Create differentiable models of complex spatiotemporal dynamics

All models are built with [JAX](https://docs.jax.dev/) for automatic differentiation, making them trainable end-to-end.

## Prerequisites

### Required Knowledge
- **Python programming** (intermediate level)
- **Basic neural networks** (understanding of gradient descent, backpropagation)
- **JAX basics** (see [JAX quickstart](https://jax.readthedocs.io/en/latest/quickstart.html))
- **NumPy** for array manipulation

### Helpful Background (not required)
- Cellular automata (e.g., Conway's Game of Life)
- Partial differential equations
- Reaction-diffusion systems

### Installation

Choose one of the following:

```bash
# CPU version
pip install -r requirements_cpu.txt

# GPU version (CUDA 12)
pip install -r requirements_gpu.txt

# Using conda
conda env create -f env_cpu.yml  # or env_gpu.yml

# Docker
docker pull alexdr1998/jax_equinox_scratch:latest
```

## Tutorial Structure

We've organized tutorials from beginner to advanced. Work through them in order:

### Level 1: Beginner (Start Here!)

| Tutorial | Topics | Time | File |
|----------|--------|------|------|
| **Tutorial 1: NCA Basics** | Setting up NCA models, training to grow patterns, visualization | 1-2 hours | `tutorials/tutorial_01_basics.ipynb` |
| **Tutorial 2: PDE Basics** | Running fixed PDEs (Gray-Scott), understanding pattern formation | 1 hour | `tutorials/tutorial_02_pde_basics.ipynb` |

**Start with Tutorial 1 if you're new to the codebase.**

### Level 2: Intermediate

| Tutorial | Topics | Time | File |
|----------|--------|------|------|
| **Tutorial 3: Advanced NCA** | Multi-species NCA, custom architectures, regeneration | 2 hours | `tutorials/tutorial_03_advanced_nca.ipynb` |
| **Tutorial 4: Neural PDEs** | Training neural-parameterized PDEs, learning reaction-diffusion systems | 2 hours | `tutorials/tutorial_04_neural_pde.ipynb` |

### Level 3: Advanced

| Tutorial | Topics | Time | File |
|----------|--------|------|------|
| **Tutorial 5: Custom Models** | Building custom update rules, new architectures, integration with JAX ecosystem | 2-3 hours | `tutorials/tutorial_05_custom_models.ipynb` |
| **Tutorial 6: Research Workflow** | Experiment organization, logging, multi-GPU training, HPC clusters | 1-2 hours | `tutorials/tutorial_06_research_workflow.ipynb` |

### Quick Reference

| Topic | File |
|-------|------|
| **Quick Start Guide** | `tutorials/QUICKSTART.md` |
| **API Reference** | `tutorials/API_REFERENCE.md` |
| **Common Errors** | `tutorials/TROUBLESHOOTING.md` |
| **Example Gallery** | `tutorials/EXAMPLES.md` |

## Learning Paths

Depending on your research goals, follow different paths:

### Path A: Neural Cellular Automata Focus
1. Tutorial 1: NCA Basics
2. Tutorial 3: Advanced NCA
3. Tutorial 5: Custom Models
4. Tutorial 6: Research Workflow

**Best for:** Researching neural models of self-organization, morphogenesis, developmental systems

### Path B: PDE Modeling Focus
1. Tutorial 2: PDE Basics
2. Tutorial 4: Neural PDEs
3. Tutorial 5: Custom Models (PDE sections)
4. Tutorial 6: Research Workflow

**Best for:** Learning physics-based pattern formation equations, chemotaxis, reaction-diffusion systems

### Path C: Complete Overview
Work through Tutorials 1-6 in order.

**Best for:** PhD students starting research in this area, comprehensive understanding

## Additional Tools

### Marimo Notebooks
The codebase includes support for [marimo](https://marimo.io/) - a reactive Python notebook alternative to Jupyter:
- `Experiments/__marimo__/` - Contains marimo session files for interactive exploration
- `marimo_utils.py` - Helper utilities for hyperparameter grid search and visualization

### Helpful Utilities
- `marimo_utils.generate_hyperparameter_combinations()` - Generate all combinations of hyperparameters for systematic experiments
- Use for parameter sweeps and ablation studies

## After the Tutorials

Once you complete the tutorials, explore:

### Demo Scripts
Located in `demo/`:
- `image_morph_demo.ipynb` - Grow images from seeds
- `image_morph_multi_species.ipynb` - Multi-channel pattern formation
- `image_morph_customise_demo.ipynb` - Customize training options
- `pde_image_classifier.ipynb` - Classification with PDEs
- `gray_scott_pde_demo.py` - Classic Gray-Scott patterns
- `chhabra_pde_demo.py` - Chhabra model patterns
- `train_nca_to_pde_demo.py` - Train NCA to mimic PDE

### Experiments Directory
Browse `Experiments/` to see complete research projects (organized by topic):
- `emoji/` - Training NCAs on emoji datasets
- `micropatterns/` - Learning biological micropatterns (individual shapes, radii, averaging)
- `multispecies/` - Multi-species and multi-stable pattern formation
- `texture/` - Texture synthesis experiments
- `pde/` - PDE learning and parameter fitting
- `nca_architecture_tests/` - Testing different NCA architectures (kaNCA, multi-scale)
- `sae/` - Sparse autoencoder experiments with NCAs
- `stability/` - Fixed point and stability analysis
- `optimizer_test/` - Comparing different optimization strategies

## Key Concepts

### Neural Cellular Automata (NCA)
- **What**: Learnable update rules applied iteratively to a grid
- **How**: Neural network processes local neighborhoods using convolutions
- **Why**: Can learn to grow complex patterns from simple seeds

### Differentiable PDEs
- **What**: Continuous differential equations discretized on a grid
- **How**: Numerical ODE solvers (via `diffrax`) on spatial derivatives
- **Why**: Model physical pattern formation processes (e.g., reaction-diffusion)

### Key JAX Patterns
All models are:
- **Equinox Modules**: Structured like PyTorch but with JAX
- **PyTrees**: Enable efficient gradient computation and multi-GPU parallelism
- **Functional**: Pure functions for JIT compilation

## Architecture Overview

```
Common/           # Shared base classes
├── model/
│   ├── abstract_model.py      # Base class for all models
│   ├── spatial_operators.py   # Differential operators (∇, ∇², etc.)
│   └── boundary.py            # Boundary conditions
├── trainer/
│   ├── abstract_data_augmenter_*.py  # Data augmentation
│   └── loss.py                       # Custom loss functions
└── utils.py                          # Helper utilities

NCA/              # Neural Cellular Automata
├── model/
│   ├── NCA_model.py           # Base NCA
│   ├── NCA_gated.py           # Gated updates
│   ├── NCA_bounded.py         # Bounded values
│   └── [8 more variants]
└── trainer/
    └── NCA_trainer.py         # Training infrastructure

PDE/              # Partial Differential Equations
├── model/
│   ├── solver/                # ODE solvers
│   ├── fixed_models/          # Classic PDEs (Gray-Scott, etc.)
│   └── reaction_diffusion_*/  # Neural parameterized PDEs
└── trainer/
    └── PDE_trainer.py         # PDE parameter training
```

## Getting Help

### If you're stuck:
1. Check `tutorials/TROUBLESHOOTING.md` for common errors
2. Review the relevant tutorial notebook
3. Look at demo scripts for working examples
4. Check test files in `tests/` for usage patterns

### Resources:
- **JAX Documentation**: https://jax.readthedocs.io/
- **Equinox Documentation**: https://docs.kidger.site/equinox/
- **Diffrax Documentation**: https://docs.kidger.site/diffrax/
- **Original Distill NCA Article**: https://distill.pub/2020/growing-ca/

## Contributing Tutorials

If you develop new tutorials or examples, please:
1. Follow the existing notebook structure
2. Include clear explanations and visualizations
3. Add them to this index
4. Submit a pull request

## Quick Start Example

Here's a minimal example to get you started:

```python
import jax
import jax.numpy as jnp
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer

# 1. Create a model
model = NCA(N_CHANNELS=16, KERNEL_STR=["ID", "LAP"])

# 2. Prepare data (e.g., target images)
data = jnp.zeros((1, 2, 4, 64, 64))  # [Batch, Time, Channels, H, W]

# 3. Train
trainer = NCA_Trainer(NCA_model=model, data=data)
trainer.train(t=64, iters=1000)

# 4. Run the trained model
x0 = jnp.zeros((4, 64, 64))  # Initial state
trajectory = model.run(100, x0)
```

More complete examples are in the tutorials!

## License

This codebase is for research and educational purposes. Please cite appropriately if you use it in your work.

---

Ready to start? Head to **Tutorial 1: NCA Basics** in `tutorials/tutorial_01_basics.ipynb`!
