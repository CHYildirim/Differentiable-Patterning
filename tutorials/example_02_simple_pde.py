"""
Example 2: Simple PDE Simulation

This script demonstrates how to simulate pattern-forming PDEs
like the Gray-Scott reaction-diffusion system.

Author: Tutorial
Date: 2024
"""

import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange
import time

# Import PDE components
from PDE.model.fixed_models.update_gray_scott import F as GrayScott
from PDE.model.fixed_models.update_chhabra import F as Chhabra
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Grid parameters
SIZE = 128                        # Grid size (128x128)
N_SPECIES = 2                     # Number of chemical species

# Gray-Scott parameters
ALPHA = 0.062                     # Feed rate
GAMMA = 0.062                     # Kill rate
DA = 0.16                         # Diffusion of species A
DB = 0.08                         # Diffusion of species B

# Solver parameters
DX = 0.5                          # Spatial resolution
DT = 0.2                          # Time step
T_MAX = 10000                     # Maximum time
N_FRAMES = 200                    # Number of frames to save

print("="*60)
print("Simple PDE Simulation Example")
print("="*60)


# ==============================================================================
# STEP 1: CREATE INITIAL CONDITIONS
# ==============================================================================

print("\n[Step 1/4] Creating initial conditions...")

# Create random initial state
key = jr.PRNGKey(int(time.time()))
x0 = jr.uniform(key, shape=(N_SPECIES, SIZE, SIZE))

print(f"✓ Created random initial state: {x0.shape}")

# Smooth the random noise to create coherent structures
op = Ops(PADDING="CIRCULAR", dx=DX, KERNEL_SCALE=3)

print("  Smoothing initial conditions...")
for i in range(5):
    x0 = op.Average(x0)

# Set up U and V species
# U: mostly 1 where V is low
# V: complement of U
x0 = x0.at[0].set(jnp.where(x0[0] > 0.51, 0.0, 1.0))
x0 = op.Average(x0)
x0 = x0.at[1].set(1 - x0[0])

print("✓ Smoothed and configured U and V species")

# Visualize initial conditions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(x0[0], cmap='viridis')
axes[0].set_title('Species U (initial)')
axes[0].axis('off')

axes[1].imshow(x0[1], cmap='plasma')
axes[1].set_title('Species V (initial)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../demo/pde_initial_conditions.png', dpi=150)
print("✓ Saved initial conditions to: ../demo/pde_initial_conditions.png")
plt.show()


# ==============================================================================
# STEP 2: DEFINE THE PDE
# ==============================================================================

print("\n[Step 2/4] Defining Gray-Scott PDE...")

# Create the Gray-Scott model
# The equations are:
#   ∂U/∂t = D_A ∇²U - UV² + α(1-U)
#   ∂V/∂t = D_V ∇²V + UV² - (α+γ)V
pde_func = GrayScott(
    PADDING="CIRCULAR",      # Periodic boundaries
    dx=DX,                   # Spatial step size
    DA=DA,                   # Diffusion coefficient for U
    DB=DB,                   # Diffusion coefficient for V
    alpha=ALPHA,             # Feed rate
    gamma=GAMMA              # Kill rate
)

print(f"✓ Created Gray-Scott model")
print(f"  Parameters: α={ALPHA}, γ={GAMMA}")
print(f"  Diffusion: D_A={DA}, D_B={DB}")


# ==============================================================================
# STEP 3: CREATE SOLVER AND SOLVE
# ==============================================================================

print("\n[Step 3/4] Creating solver and solving PDE...")
print(f"  Time range: 0 to {T_MAX}")
print(f"  Number of frames: {N_FRAMES}")
print(f"  This may take a minute...\n")

# Create the PDE solver
solver = PDE_solver(
    pde_func,
    dt=DT,                   # Time step
    SOLVER="heun",           # Numerical method (heun is a good default)
    ADAPTIVE=False           # Use fixed time stepping
)

# Solve the PDE
start_time = time.time()
T, Y = solver(
    ts=jnp.linspace(0, T_MAX, N_FRAMES),  # Time points to evaluate
    y0=x0                                   # Initial condition
)
elapsed = time.time() - start_time

print(f"✓ Solved PDE in {elapsed:.2f} seconds")
print(f"  Solution shape: {Y.shape}")
print(f"  (Time, Species, Height, Width)")


# ==============================================================================
# STEP 4: VISUALIZE RESULTS
# ==============================================================================

print("\n[Step 4/4] Visualizing results...")

# Show evolution over time
frames_to_show = [0, 40, 80, 120, 160, 199]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, frame_idx in enumerate(frames_to_show):
    axes[i].imshow(Y[frame_idx, 0], cmap='viridis')
    axes[i].set_title(f't = {T[frame_idx]:.1f}')
    axes[i].axis('off')

plt.suptitle('Gray-Scott Pattern Evolution (Species U)', fontsize=16)
plt.tight_layout()
plt.savefig('../demo/pde_evolution.png', dpi=150)
print("✓ Saved evolution to: ../demo/pde_evolution.png")
plt.show()

# Compare U and V at final time
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Y[-1, 0], cmap='viridis')
axes[0].set_title('Species U (final)')
axes[0].axis('off')

axes[1].imshow(Y[-1, 1], cmap='plasma')
axes[1].set_title('Species V (final)')
axes[1].axis('off')

# Combined view (U=red, V=green)
rgb = jnp.stack([Y[-1, 0], Y[-1, 1], jnp.zeros_like(Y[-1, 0])], axis=-1)
axes[2].imshow(rgb)
axes[2].set_title('Combined (U=red, V=green)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('../demo/pde_final_state.png', dpi=150)
print("✓ Saved final state to: ../demo/pde_final_state.png")
plt.show()

# Create animation
print("\nCreating animation...")
my_animate(Y[:, :1], clip=False)  # Animate species U
print("✓ Animation displayed")


# ==============================================================================
# BONUS: EXPLORE PARAMETER SPACE
# ==============================================================================

print("\n[Bonus] Exploring different parameter regimes...")

# Different Gray-Scott parameter sets produce different patterns
parameter_sets = {
    "Spots": {"alpha": 0.062, "gamma": 0.061},
    "Stripes": {"alpha": 0.046, "gamma": 0.063},
    "Waves": {"alpha": 0.014, "gamma": 0.054},
    "Mixed": {"alpha": 0.034, "gamma": 0.056},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (name, params) in enumerate(parameter_sets.items()):
    print(f"  Running {name} configuration...")

    # Create PDE with these parameters
    pde = GrayScott(
        PADDING="CIRCULAR",
        dx=DX,
        DA=DA,
        DB=DB,
        alpha=params["alpha"],
        gamma=params["gamma"]
    )
    solver = PDE_solver(pde, dt=DT)

    # Solve
    T_param, Y_param = solver(
        ts=jnp.linspace(0, T_MAX, 100),
        y0=x0
    )

    # Plot final state
    axes[idx].imshow(Y_param[-1, 0], cmap='viridis')
    axes[idx].set_title(f'{name}: α={params["alpha"]}, γ={params["gamma"]}')
    axes[idx].axis('off')

plt.suptitle('Gray-Scott: Different Parameter Regimes', fontsize=16)
plt.tight_layout()
plt.savefig('../demo/pde_parameter_exploration.png', dpi=150)
print("✓ Saved parameter exploration to: ../demo/pde_parameter_exploration.png")
plt.show()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*60)
print("SUCCESS! Summary:")
print("="*60)
print(f"✓ Simulated Gray-Scott PDE on {SIZE}x{SIZE} grid")
print(f"✓ Generated {N_FRAMES} frames from t=0 to t={T_MAX}")
print(f"✓ Explored 4 different parameter regimes")
print(f"✓ All visualizations saved to demo/")
print("\nNext steps:")
print("  1. Try different ALPHA and GAMMA values")
print("  2. Try the Chhabra model (see code below)")
print("  3. Create your own PDE (see tutorial_02_pde_basics.ipynb)")
print("  4. Train an NCA to mimic the PDE patterns")
print("="*60)


# ==============================================================================
# EXERCISES FOR STUDENTS
# ==============================================================================

"""
EXERCISES TO TRY:

1. Parameter Exploration:
   Systematically explore the (α, γ) parameter space:
   - α ∈ [0.01, 0.08]
   - γ ∈ [0.05, 0.07]
   Create a grid of results to see the full range of patterns

2. Different Initial Conditions:
   Try starting from:
   - Single localized spot
   - Periodic stripes
   - Different noise frequencies

   initial_spot = jnp.zeros((2, SIZE, SIZE))
   initial_spot = initial_spot.at[1, SIZE//2-5:SIZE//2+5, SIZE//2-5:SIZE//2+5].set(1.0)

3. Chhabra Model:
   Uncomment this code to try the Chhabra PDE instead:

   pde = Chhabra(
       PADDING="CIRCULAR",
       dx=0.5,
       D1=1.0,
       D2=10.0,
       k1=0.5,
       k2=1.0,
       k3=1.0
   )

4. Different Solvers:
   Try SOLVER="euler", "heun", "dopri5"
   Compare accuracy and speed

5. Adaptive Stepping:
   Set ADAPTIVE=True and experiment with ATOL and RTOL

   solver = PDE_solver(pde, dt=0.1, ADAPTIVE=True, ATOL=1e-6, RTOL=1e-3)

6. Multi-Scale Patterns:
   What happens if you change dx? Try dx=0.25 (finer) or dx=1.0 (coarser)

7. Create Custom PDE:
   Write your own pattern-forming PDE:

   class MyPDE(eqx.Module):
       op: Ops

       def __init__(self):
           self.op = Ops(PADDING="CIRCULAR", dx=0.5)

       def __call__(self, t, y, args):
           # Your PDE here!
           # For example: simple diffusion
           return 0.1 * self.op.Lap(y)
"""
