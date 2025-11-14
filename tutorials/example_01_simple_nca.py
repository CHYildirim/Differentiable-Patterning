"""
Example 1: Simple NCA Training

This script demonstrates the minimal code needed to train an NCA
to grow a pattern from a seed.

Author: Tutorial
Date: 2024
"""

import sys
sys.path.append('..')  # Add parent directory to path

import jax
import jax.numpy as jnp
import optax
from einops import rearrange
import matplotlib.pyplot as plt

# Import NCA components
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca import DataAugmenter
from Common.dataloader.emoji import load_emoji_sequence


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model hyperparameters
N_CHANNELS = 16                    # Number of channels per cell
KERNEL_STR = ["ID", "LAP"]        # Spatial perception kernels
ACTIVATION = jax.nn.relu          # Nonlinearity
PADDING = "CIRCULAR"               # Boundary condition
FIRE_RATE = 0.5                   # Stochastic update probability

# Training hyperparameters
TRAINING_STEPS = 500              # How many gradient updates
NCA_STEPS = 64                    # How many NCA iterations per update
LEARNING_RATE = 1e-4              # Initial learning rate
DOWNSAMPLE = 4                    # Reduce image size for speed

# Paths
DATA_PATH = "../demo/demo_data/"
MODEL_SAVE_PATH = "../demo/models/simple_nca_example.eqx"
LOG_DIR = "../demo/logs/"

print("="*60)
print("Simple NCA Training Example")
print("="*60)


# ==============================================================================
# STEP 1: CREATE THE MODEL
# ==============================================================================

print("\n[Step 1/5] Creating NCA model...")

model = NCA(
    N_CHANNELS=N_CHANNELS,
    KERNEL_STR=KERNEL_STR,
    ACTIVATION=ACTIVATION,
    PADDING=PADDING,
    FIRE_RATE=FIRE_RATE
)

print(f"✓ Created NCA with {N_CHANNELS} channels")
print(f"  Kernels: {KERNEL_STR}")
print(f"  Fire rate: {FIRE_RATE}")


# ==============================================================================
# STEP 2: LOAD AND PREPARE DATA
# ==============================================================================

print("\n[Step 2/5] Loading training data...")

# Load the target image
# We load it twice to create a sequence of [initial, target]
data = load_emoji_sequence(
    ["crab.png", "crab.png"],
    impath_emojis=DATA_PATH,
    downsample=DOWNSAMPLE
)

print(f"✓ Loaded data with shape: {data.shape}")
print(f"  (Batch, Time, Channels, Height, Width)")

# Create initial seed condition (small center square)
initial_condition = jnp.array(data[:, :1])  # Take first timestep
W, H = initial_condition.shape[-2:]

# Zero out everything except 3x3 center region
initial_condition = initial_condition.at[0, 0, :, :W//2-2].set(0)  # Left half
initial_condition = initial_condition.at[0, 0, :, W//2+1:].set(0)   # Right half
initial_condition = initial_condition.at[0, 0, :, :, :H//2-2].set(0)  # Top half
initial_condition = initial_condition.at[0, 0, :, :, H//2+1:].set(0)   # Bottom half

# Combine seed and target
data = jnp.concatenate([initial_condition, data], axis=1)
print(f"✓ Created seed condition of size 3x3")
print(f"  Final data shape: {data.shape}")


# ==============================================================================
# STEP 3: CREATE TRAINER
# ==============================================================================

print("\n[Step 3/5] Setting up trainer...")

# Define a simple data augmentation class
class SimpleAugmenter(DataAugmenter):
    """Add padding around the image."""
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.pad(data, 10)  # Add 10 pixel border
        self.save_data(data)
        return None


# Create the trainer object
trainer = NCA_Trainer(
    NCA_model=model,
    data=data,
    DATA_AUGMENTER=SimpleAugmenter,
    model_filename="simple_nca_example",
    MODEL_DIRECTORY="../demo/models/",
    LOG_DIRECTORY=LOG_DIR
)

print("✓ Trainer created")


# ==============================================================================
# STEP 4: TRAIN THE MODEL
# ==============================================================================

print("\n[Step 4/5] Training model...")
print(f"  Steps: {TRAINING_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  This may take a few minutes...\n")

# Set up optimizer with learning rate decay
schedule = optax.exponential_decay(
    init_value=LEARNING_RATE,
    transition_steps=TRAINING_STEPS,
    decay_rate=0.99  # Decay by 1% each step
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
    optax.adam(schedule)              # Adam optimizer with schedule
)

# Train!
trainer.train(
    t=NCA_STEPS,                      # NCA steps per training iteration
    iters=TRAINING_STEPS,             # Total training iterations
    optimiser=optimizer,
    WARMUP=10,                        # Warmup iterations before full training
    LOOP_AUTODIFF="lax",             # Use standard autodiff (faster)
    REGULARISER_COEFFS={              # Regularization terms
        "intermediate_state": 1.0,    # Match intermediate frames
        "contigous_growth": 1.0,      # Penalize discontinuous growth
    },
    wandb_args={                      # Logging configuration
        "project": "NCA_Tutorial",
        "name": "simple_example",
        "group": "examples"
    },
    LOG_EVERY=50                      # Log every 50 steps
)

print("\n✓ Training complete!")


# ==============================================================================
# STEP 5: TEST AND VISUALIZE
# ==============================================================================

print("\n[Step 5/5] Testing trained model...")

# Get the trained model
trained_model = trainer.NCA_model

# Save it
trained_model.save(MODEL_SAVE_PATH)
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

# Get initial seed from augmented data
x0 = trainer.DATA_AUGMENTER.return_saved_data()[0][0]

# Run the NCA
trajectory = trained_model.run(NCA_STEPS, x0)
print(f"✓ Generated trajectory with shape: {trajectory.shape}")

# Visualize results
print("\nGenerating visualization...")

# Show every 8th frame horizontally
frames_to_show = trajectory[::8]
vis = rearrange(frames_to_show[:, :3], "T C H W -> H (T W) C")

plt.figure(figsize=(20, 5))
plt.imshow(vis)
plt.title("NCA Growth Over Time (every 8 steps)")
plt.axis('off')
plt.tight_layout()
plt.savefig("../demo/simple_nca_result.png", dpi=150)
print("✓ Saved visualization to: ../demo/simple_nca_result.png")
plt.show()

# Compare seed vs final state
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Initial seed
seed_vis = rearrange(trajectory[0, :3], "C H W -> H W C")
axes[0].imshow(seed_vis)
axes[0].set_title("Initial Seed (t=0)")
axes[0].axis('off')

# Final state
final_vis = rearrange(trajectory[-1, :3], "C H W -> H W C")
axes[1].imshow(final_vis)
axes[1].set_title(f"Final State (t={NCA_STEPS})")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("../demo/simple_nca_comparison.png", dpi=150)
print("✓ Saved comparison to: ../demo/simple_nca_comparison.png")
plt.show()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*60)
print("SUCCESS! Summary:")
print("="*60)
print(f"✓ Trained NCA with {N_CHANNELS} channels")
print(f"✓ {TRAINING_STEPS} training iterations completed")
print(f"✓ Model saved to {MODEL_SAVE_PATH}")
print(f"✓ Visualizations saved")
print("\nNext steps:")
print("  1. Try changing N_CHANNELS, KERNEL_STR, or FIRE_RATE")
print("  2. Train for more steps (increase TRAINING_STEPS)")
print("  3. Try different images from demo/demo_data/")
print("  4. See tutorials/ for more advanced examples")
print("="*60)


# ==============================================================================
# EXERCISES FOR STUDENTS
# ==============================================================================

"""
EXERCISES TO TRY:

1. Different Kernels:
   Change KERNEL_STR to ["ID", "GRAD", "LAP"] or ["ID", "AVG"]
   How does this affect the patterns?

2. More Channels:
   Set N_CHANNELS = 32
   Does this improve quality? How does it affect training time?

3. Fire Rate:
   Try FIRE_RATE = 0.0 (deterministic) or 1.0 (fully stochastic)
   What difference does it make?

4. Longer Training:
   Set TRAINING_STEPS = 2000
   Does longer training help?

5. Different Images:
   Replace "crab.png" with other images from demo/demo_data/
   Try both simple and complex patterns

6. Damage Test:
   After training, damage the final state and see if it regenerates:

   x_damaged = trajectory[-1].at[:, :, :W//2].set(0)
   regen_trajectory = trained_model.run(100, x_damaged)

   Can it fix itself?
"""
