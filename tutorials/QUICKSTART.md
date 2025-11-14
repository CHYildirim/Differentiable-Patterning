# Quick Start Guide

Get up and running with Differentiable-Patterning in 5 minutes!

## Installation

```bash
# CPU version (recommended for learning)
pip install -r requirements_cpu.txt

# OR GPU version (for serious training)
pip install -r requirements_gpu.txt
```

## Hello World: Growing a Pattern with NCA

```python
import jax
import jax.numpy as jnp
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from NCA.trainer.data_augmenter_nca import DataAugmenter

# 1. Create an NCA model
model = NCA(
    N_CHANNELS=16,
    KERNEL_STR=["ID", "LAP"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5
)

# 2. Load data
data = load_emoji_sequence(
    ["crab.png", "crab.png"],
    impath_emojis="demo/demo_data/",
    downsample=4
)

# 3. Create a seed (small initial condition)
initial = jnp.array(data[:, :1])
W, H = initial.shape[-2:]
initial = initial.at[0, 0, :, :W//2-2].set(0)
initial = initial.at[0, 0, :, W//2+1:].set(0)
initial = initial.at[0, 0, :, :, :H//2-2].set(0)
initial = initial.at[0, 0, :, :, H//2+1:].set(0)

data = jnp.concatenate([initial, data], axis=1)

# 4. Train
trainer = NCA_Trainer(
    NCA_model=model,
    data=data,
    DATA_AUGMENTER=DataAugmenter,
    model_filename="my_first_nca"
)

trainer.train(t=64, iters=1000)

# 5. Run the trained model
x0 = trainer.DATA_AUGMENTER.return_saved_data()[0][0]
trajectory = trainer.NCA_model.run(64, x0)

print("Success! Your NCA has learned to grow a pattern!")
```

## Hello World: Simulating a PDE

```python
import jax.numpy as jnp
import jax.random as jr
from PDE.model.fixed_models.update_gray_scott import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops

# 1. Create initial conditions
SIZE = 128
key = jr.PRNGKey(42)
x0 = jr.uniform(key, shape=(2, SIZE, SIZE))

# Smooth them
op = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)
for i in range(5):
    x0 = op.Average(x0)
x0 = x0.at[0].set(jnp.where(x0[0] > 0.51, 0.0, 1.0))
x0 = op.Average(x0)
x0 = x0.at[1].set(1 - x0[0])

# 2. Define the PDE
func = F(
    PADDING="CIRCULAR",
    dx=0.5,
    alpha=0.0623,
    gamma=0.06268
)

# 3. Create solver and solve
solver = PDE_solver(func, dt=0.2)
T, Y = solver(ts=jnp.linspace(0, 10000, 200), y0=x0)

print("Success! You've simulated pattern formation!")
print(f"Solution shape: {Y.shape}")
```

## Common Tasks

### Load and visualize results

```python
from einops import rearrange
import matplotlib.pyplot as plt

# For NCA trajectory
trajectory_rgb = rearrange(trajectory[::8, :3], "T C H W -> H (T W) C")
plt.imshow(trajectory_rgb)
plt.show()

# For PDE solution
plt.imshow(Y[-1, 0], cmap='viridis')
plt.show()
```

### Save and load models

```python
# Save
model.save("path/to/model.eqx")

# Load
model = NCA(N_CHANNELS=16, KERNEL_STR=["ID", "LAP"])
model = model.load("path/to/model.eqx")
```

### Change devices (GPU/CPU)

```python
# JAX automatically uses available GPUs
# To force CPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Check what's available:
import jax
print(jax.devices())
```

## Next Steps

1. **Complete beginner tutorials:**
   - `tutorials/tutorial_01_basics.ipynb` - NCA fundamentals
   - `tutorials/tutorial_02_pde_basics.ipynb` - PDE simulations

2. **Explore demos:**
   - `demo/image_morph_demo.ipynb`
   - `demo/gray_scott_pde_demo.py`

3. **Read full documentation:**
   - `TUTORIALS.md` - Full tutorial index
   - `README.md` - Project overview

## Get Help

- **Errors?** Check `tutorials/TROUBLESHOOTING.md`
- **API questions?** See `tutorials/API_REFERENCE.md`
- **Concepts unclear?** Start with Tutorial 1

## Minimum Working Example (Copy-Paste Ready)

Save as `test.py` and run with `python test.py`:

```python
import jax
import jax.numpy as jnp
from NCA.model.NCA_model import NCA

# Create model
model = NCA(N_CHANNELS=16, KERNEL_STR=["ID", "LAP"])

# Create initial state
x0 = jnp.zeros((16, 64, 64))
x0 = x0.at[:3, 31:33, 31:33].set(1.0)

# Run
trajectory = model.run(100, x0)
print(f"Generated trajectory with shape: {trajectory.shape}")
print("Success!")
```

Happy patterning! 🦀
