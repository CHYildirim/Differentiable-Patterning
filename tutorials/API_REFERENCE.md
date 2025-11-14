# API Reference

Quick reference for the most commonly used classes and functions.

## Table of Contents
- [NCA Models](#nca-models)
- [NCA Training](#nca-training)
- [PDE Models](#pde-models)
- [PDE Solving](#pde-solving)
- [Spatial Operators](#spatial-operators)
- [Data Loading](#data-loading)
- [Utilities](#utilities)

---

## NCA Models

### NCA (Base Model)

```python
from NCA.model.NCA_model import NCA

model = NCA(
    N_CHANNELS=16,              # Number of channels per cell
    KERNEL_STR=["ID", "LAP"],   # Spatial kernels to use
    ACTIVATION=jax.nn.relu,     # Activation function
    PADDING="CIRCULAR",         # Boundary conditions
    FIRE_RATE=0.5,             # Update probability [0, 1]
    KERNEL_SCALE=1,            # Kernel size scale
    USE_BIAS=True,             # Use bias in linear layers
    HIDDEN_CHANNELS=128        # Hidden layer size
)
```

**Methods:**
- `model.run(T, x0)` - Run NCA for T steps from initial state x0
- `model.save(path)` - Save model to file
- `model.load(path)` - Load model from file
- `model.partition()` - Split into trainable/static parameters
- `model.combine(trainable, static)` - Recombine parameters

**Kernel Options:**
- `"ID"` - Identity (cell's own value)
- `"LAP"` - Laplacian (difference from neighbors)
- `"GRAD"` - Gradient (directional derivatives)
- `"AVG"` - Average of neighbors
- `"ID_LAP"` - Combined identity and Laplacian

**Padding Options:**
- `"CIRCULAR"` - Periodic boundaries (torus)
- `"ZEROS"` - Zero padding
- `"REFLECT"` - Mirror boundaries
- `"REPLICATE"` - Extend edge values

### NCA Variants

All have the same API as base NCA, with additional parameters:

```python
from NCA.model.NCA_gated import gNCA        # Gated updates
from NCA.model.NCA_bounded import bNCA      # Bounded values
from NCA.model.NCA_gated_bounded import gbNCA  # Both features
from NCA.model.NCA_higher_order import hNCA    # Higher-order interactions
from NCA.model.NCA_multi_step import mNCA      # Multi-scale
from NCA.model.NCA_attention import aNCA       # Attention-based
```

**Example:**
```python
gated_model = gNCA(
    N_CHANNELS=16,
    KERNEL_STR=["ID", "LAP"],
    GATE_THRESHOLD=0.5  # Additional parameter for gating
)
```

---

## NCA Training

### NCA_Trainer

```python
from NCA.trainer.NCA_trainer import NCA_Trainer

trainer = NCA_Trainer(
    NCA_model=model,                      # The NCA model to train
    data=data,                            # Training data (Batch, Time, Channels, H, W)
    DATA_AUGMENTER=DataAugmenter,        # Data augmentation class
    model_filename="my_model",            # Name for saving
    MODEL_DIRECTORY="models/",            # Where to save models
    LOG_DIRECTORY="logs/",                # Where to save logs
    SHARDING=None                         # For multi-GPU training
)
```

**Training Method:**
```python
trainer.train(
    t=64,                          # Number of NCA steps per iteration
    iters=1000,                    # Total training iterations
    optimiser=optax.adam(1e-3),    # Optimizer
    WARMUP=10,                     # Warmup iterations
    LOOP_AUTODIFF="lax",          # "lax" or "checkpointed"
    CLEAR_CACHE_EVERY=500,        # Clear JAX cache frequency
    REGULARISER_COEFFS={           # Regularization terms
        "intermediate_state": 1.0,
        "contigous_growth": 1.0,
        "living_cells": 0.0
    },
    LOG_EVERY=50,                  # Logging frequency
    wandb_args={                   # Weights & Biases logging
        "project": "MyProject",
        "name": "experiment_1",
        "group": "experiments"
    }
)
```

**Regularization Options:**
- `intermediate_state`: Match intermediate frames during growth
- `contigous_growth`: Penalize discontinuous growth
- `living_cells`: Penalize too many active cells
- `gradient_norm`: Penalize large parameter gradients

### Data Augmenter

```python
from NCA.trainer.data_augmenter_nca import DataAugmenter

class MyAugmenter(DataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()

        # Apply transformations
        data = self.duplicate_batches(data, n=4)
        data = self.pad(data, padding=10)
        data = self.random_rotate(data)
        data = self.random_scale(data, scale_range=(0.8, 1.2))

        self.save_data(data)
        return None
```

**Available Methods:**
- `duplicate_batches(data, n)` - Duplicate for more batch diversity
- `pad(data, padding)` - Add padding around images
- `random_rotate(data)` - Random rotations
- `random_scale(data, scale_range)` - Random scaling
- `add_hidden_channels(data, n_channels)` - Add hidden channels

---

## PDE Models

### Fixed PDE Models

```python
from PDE.model.fixed_models.update_gray_scott import F as GrayScott
from PDE.model.fixed_models.update_chhabra import F as Chhabra
from PDE.model.fixed_models.update_cahn_hilliard import F as CahnHilliard
from PDE.model.fixed_models.update_keller_segel import F as KellerSegel

# Gray-Scott
pde = GrayScott(
    PADDING="CIRCULAR",
    dx=0.5,              # Spatial resolution
    DA=0.16,             # Diffusion coefficient for species A
    DB=0.08,             # Diffusion coefficient for species B
    alpha=0.060,         # Feed rate
    gamma=0.062          # Kill rate
)

# Chhabra
pde = Chhabra(
    PADDING="CIRCULAR",
    dx=0.5,
    D1=1.0,              # Diffusion for species 1
    D2=10.0,             # Diffusion for species 2
    k1=0.5, k2=1.0, k3=1.0  # Reaction rates
)
```

### Neural Parameterized PDEs

```python
from PDE.model.reaction_diffusion_advection.update import F as RDA
from PDE.model.reaction_diffusion_chemotaxis.update import F as RDC

# Reaction-Diffusion-Advection
pde = RDA(
    N_CHANNELS=2,
    PADDING="CIRCULAR",
    dx=0.5,
    TERMS=["reaction", "diffusion", "advection"],  # Which terms to include
    HIDDEN=64,           # Neural network hidden size
    LAYERS=2,            # Network depth
    INIT_SCALE=1e-3     # Parameter initialization scale
)

# Reaction-Diffusion-Chemotaxis
pde = RDC(
    N_CELLS=2,           # Number of cell types
    N_SIGNALS=1,         # Number of chemical signals
    PADDING="CIRCULAR",
    dx=0.5,
    INCLUDE_CHEMOTAXIS=True,
    INCLUDE_HAPTOTAXIS=False
)
```

---

## PDE Solving

### PDE_solver

```python
from PDE.model.solver.semidiscrete_solver import PDE_solver

solver = PDE_solver(
    func,                    # PDE right-hand side (eqx.Module)
    dt=0.1,                 # Time step
    SOLVER="heun",          # Solver method
    ADAPTIVE=False,         # Use adaptive stepping?
    ATOL=1e-6,             # Absolute tolerance (if adaptive)
    RTOL=1e-3              # Relative tolerance (if adaptive)
)
```

**Solver Options:**
- `"euler"` - Forward Euler (fast, less accurate)
- `"heun"` - Heun's method (good balance)
- `"dopri5"` - Dormand-Prince (accurate, adaptive)
- `"dopri8"` - Higher order (very accurate)
- `"tsit5"` - Tsitouras 5(4) (efficient adaptive)

**Usage:**
```python
T, Y = solver(
    ts=jnp.linspace(0, 100, 200),  # Time points to evaluate
    y0=initial_state                # Initial condition
)
```

**Returns:**
- `T`: Time points (shape: [n_timepoints])
- `Y`: Solutions (shape: [n_timepoints, n_channels, height, width])

---

## Spatial Operators

### Ops (Spatial Operators)

```python
from Common.model.spatial_operators import Ops

op = Ops(
    PADDING="CIRCULAR",     # Boundary condition
    dx=1.0,                # Spatial step size
    KERNEL_SCALE=1         # Kernel size multiplier
)
```

**Methods:**

```python
# Input x has shape: (Channels, Height, Width)

laplacian = op.Lap(x)           # ∇²x (shape: [C, H, W])
gradient = op.Grad(x)           # ∇x (shape: [C, 2, H, W])
divergence = op.Div(v)          # ∇·v (v has shape [C, 2, H, W])
averaged = op.Average(x)        # Smooth with averaging
grad_mag = op.GradMag(x)       # |∇x|
grad_laplacian = op.GradLap(x) # ∇(∇²x)
```

**Example:**
```python
# Compute diffusion: D*∇²u
D = 0.1
diffusion = D * op.Lap(u)

# Compute advection: -v·∇u
advection = -op.Div(v * op.Grad(u))
```

---

## Data Loading

### Emoji Loader

```python
from Common.dataloader.emoji import load_emoji_sequence

data = load_emoji_sequence(
    ["crab.png", "octopus.png"],     # List of image files
    impath_emojis="demo/demo_data/", # Directory path
    downsample=4,                     # Downsampling factor
    SIMPLE_RGBA=True                  # Use RGBA (not RGBA+alpha)
)
```

**Returns:** Array of shape `(Batch, Time, Channels, Height, Width)`

### Texture Loader

```python
from Common.dataloader.texture import load_texture

data = load_texture(
    "texture.png",
    impath="path/to/textures/",
    size=128,               # Resize to this size
    crop=True              # Center crop?
)
```

### Micropattern Loader

```python
from Common.dataloader.micropatterns import load_micropatterns

data = load_micropatterns(
    pattern_type="spots",   # "spots", "stripes", "labyrinth"
    n_samples=10,
    size=128
)
```

---

## Utilities

### Visualization

```python
from Common.utils import my_animate, save_to_video

# Create animation
my_animate(
    trajectory,        # Shape: (Time, Channels, H, W)
    clip=True,        # Clip values to [0, 1]?
    fps=30            # Frames per second
)

# Save video
save_to_video(
    trajectory,
    filename="output.mp4",
    fps=30
)
```

### Using einops for reshaping

```python
from einops import rearrange, repeat, reduce

# Show time evolution horizontally
vis = rearrange(traj[:, :3], "T C H W -> H (T W) C")

# Stack multiple samples
vis = rearrange(batch, "B C H W -> (B H) W C")

# Spatially subsample
vis = reduce(img, "C (H h) (W w) -> C H W", "mean", h=2, w=2)
```

---

## Common Patterns

### Training Loop

```python
import optax

# 1. Create model
model = NCA(N_CHANNELS=16, KERNEL_STR=["ID", "LAP"])

# 2. Prepare data
data = load_emoji_sequence(["crab.png", "crab.png"], downsample=4)

# 3. Create trainer
trainer = NCA_Trainer(model, data)

# 4. Set up optimizer
schedule = optax.exponential_decay(1e-4, 1000, 0.99)
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(schedule)
)

# 5. Train
trainer.train(t=64, iters=1000, optimiser=opt)

# 6. Save
trainer.NCA_model.save("trained_model.eqx")
```

### PDE Simulation

```python
# 1. Create initial conditions
x0 = jr.uniform(key, shape=(2, 128, 128))

# 2. Define PDE
pde = GrayScott(PADDING="CIRCULAR", dx=0.5, alpha=0.06, gamma=0.062)

# 3. Create solver
solver = PDE_solver(pde, dt=0.2)

# 4. Solve
T, Y = solver(ts=jnp.linspace(0, 1000, 200), y0=x0)

# 5. Visualize
my_animate(Y[:, :1])
```

### Custom PDE

```python
import equinox as eqx

class MyPDE(eqx.Module):
    D: float
    op: Ops

    def __init__(self, D=1.0):
        self.D = D
        self.op = Ops(PADDING="CIRCULAR", dx=1.0)

    def __call__(self, t, y, args):
        # Define ∂y/∂t
        diffusion = self.D * self.op.Lap(y)
        reaction = y * (1 - y)  # Logistic growth
        return diffusion + reaction

# Use it
pde = MyPDE(D=0.1)
solver = PDE_solver(pde, dt=0.1)
T, Y = solver(ts=jnp.linspace(0, 100, 100), y0=x0)
```

---

## Type Signatures

For reference, here are the expected array shapes:

```python
# NCA
x0: (Channels, Height, Width)
trajectory: (Time, Channels, Height, Width)
data: (Batch, Time, Channels, Height, Width)

# PDE
y0: (Channels, Height, Width)
solution: (Time, Channels, Height, Width)

# Spatial operators
input: (Channels, Height, Width)
gradient_output: (Channels, 2, Height, Width)  # 2 for x and y components
laplacian_output: (Channels, Height, Width)
```

---

## Configuration Templates

### Fast Prototyping

```python
# Small, fast configuration for testing
model = NCA(N_CHANNELS=8, KERNEL_STR=["ID", "LAP"])
trainer.train(t=32, iters=100)
SIZE = 64
DOWNSAMPLE = 8
```

### Production Training

```python
# Large, slow configuration for final results
model = NCA(N_CHANNELS=32, KERNEL_STR=["ID", "GRAD", "LAP"])
trainer.train(t=128, iters=5000, LOOP_AUTODIFF="checkpointed")
SIZE = 256
DOWNSAMPLE = 1
```

---

For more details, see the source code and docstrings in each module!
