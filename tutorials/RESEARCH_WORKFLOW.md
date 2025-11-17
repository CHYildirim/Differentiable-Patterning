# Research Workflow Guide

How to organize experiments and conduct research with this codebase.

## Table of Contents
- [Project Organization](#project-organization)
- [Setting Up Experiments](#setting-up-experiments)
- [Version Control Best Practices](#version-control-best-practices)
- [Logging and Tracking](#logging-and-tracking)
- [Running on HPC Clusters](#running-on-hpc-clusters)
- [Reproducibility](#reproducibility)
- [Writing Papers](#writing-papers)

---

## Project Organization

### Recommended Directory Structure

The codebase organizes experiments by topic. Follow this pattern for your work:

```
Differentiable-Patterning/
├── Experiments/
│   ├── emoji/                   # Existing: emoji training
│   ├── micropatterns/           # Existing: biological micropatterns
│   ├── multispecies/            # Existing: multi-species systems
│   ├── texture/                 # Existing: texture synthesis
│   ├── pde/                     # Existing: PDE experiments
│   ├── nca_architecture_tests/  # Existing: architecture comparisons
│   ├── sae/                     # Existing: sparse autoencoders
│   ├── stability/               # Existing: stability analysis
│   ├── optimizer_test/          # Existing: optimizer comparisons
│   └── my_project/              # Your new research project
│       ├── README.md           # Project description
│       ├── config/             # Configuration files
│       │   ├── base_config.py
│       │   └── experiment_configs.py
│       ├── scripts/            # Experiment scripts
│       │   ├── train_baseline.py
│       │   ├── train_variant_1.py
│       │   └── analysis.py
│       ├── notebooks/          # Analysis notebooks
│       │   ├── visualize_results.ipynb
│       │   └── generate_figures.ipynb
│       ├── data/              # Project-specific data
│       ├── models/            # Saved models
│       ├── logs/              # Training logs
│       ├── results/           # Processed results
│       └── figures/           # Generated figures
```

### Creating a New Project

Choose an appropriate subdirectory based on your research topic, or create a new one:

```bash
cd Experiments/

# Option 1: Add to existing category
cd micropatterns/  # or emoji/, pde/, etc.

# Option 2: Create new category for your project
mkdir my_project
cd my_project

# Create structure
mkdir -p {config,scripts,notebooks,data,models,logs,results,figures}

# Create README
cat > README.md << 'EOF'
# Project: [Your Project Name]

## Objective
[Describe what you're trying to accomplish]

## Hypothesis
[What do you expect to find?]

## Methods
[Brief description of approach]

## Results
[Key findings - update as you go]

## Next Steps
[What to do next]
EOF
```

---

## Setting Up Experiments

### 1. Define Configuration

Create `config/base_config.py`:

```python
"""Base configuration for all experiments."""

import jax

# Model configuration
MODEL_CONFIG = {
    "N_CHANNELS": 16,
    "KERNEL_STR": ["ID", "GRAD", "LAP"],
    "ACTIVATION": jax.nn.relu,
    "PADDING": "CIRCULAR",
    "FIRE_RATE": 0.5,
}

# Training configuration
TRAIN_CONFIG = {
    "t": 64,                    # NCA steps per iteration
    "iters": 2000,              # Total iterations
    "learning_rate": 1e-4,
    "batch_size": 4,
    "warmup": 10,
}

# Data configuration
DATA_CONFIG = {
    "downsample": 4,
    "size": 128,
    "padding": 10,
}

# Paths
PATHS = {
    "data": "data/",
    "models": "models/",
    "logs": "logs/",
    "results": "results/",
}
```

Create `config/experiment_configs.py`:

```python
"""Specific experiment configurations."""

from config.base_config import MODEL_CONFIG, TRAIN_CONFIG

# Experiment 1: Baseline
EXP1_CONFIG = {
    "name": "baseline",
    "model": MODEL_CONFIG.copy(),
    "train": TRAIN_CONFIG.copy(),
    "description": "Standard NCA with default parameters"
}

# Experiment 2: High channel count
EXP2_CONFIG = {
    "name": "high_channels",
    "model": {**MODEL_CONFIG, "N_CHANNELS": 32},
    "train": TRAIN_CONFIG.copy(),
    "description": "NCA with more channels for increased capacity"
}

# Experiment 3: Different kernels
EXP3_CONFIG = {
    "name": "simple_kernels",
    "model": {**MODEL_CONFIG, "KERNEL_STR": ["ID", "LAP"]},
    "train": TRAIN_CONFIG.copy(),
    "description": "NCA with only identity and Laplacian"
}

# Dictionary of all experiments
EXPERIMENTS = {
    "exp1": EXP1_CONFIG,
    "exp2": EXP2_CONFIG,
    "exp3": EXP3_CONFIG,
}
```

### 2. Create Experiment Script

Create `scripts/run_experiment.py`:

```python
"""Run experiments from configuration."""

import sys
sys.path.append('../..')

import argparse
import jax
import jax.numpy as jnp
import optax
from pathlib import Path

from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from NCA.trainer.data_augmenter_nca import DataAugmenter

from config.experiment_configs import EXPERIMENTS
from config.base_config import DATA_CONFIG, PATHS


def setup_directories():
    """Create necessary directories."""
    for path in PATHS.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data():
    """Load training data."""
    data = load_emoji_sequence(
        ["crab.png", "crab.png"],
        impath_emojis="../../demo/demo_data/",
        downsample=DATA_CONFIG["downsample"]
    )
    # Create seed
    initial = jnp.array(data[:, :1])
    W, H = initial.shape[-2:]
    initial = initial.at[0, 0, :, :W//2-2].set(0)
    initial = initial.at[0, 0, :, W//2+1:].set(0)
    initial = initial.at[0, 0, :, :, :H//2-2].set(0)
    initial = initial.at[0, 0, :, :, H//2+1:].set(0)
    data = jnp.concatenate([initial, data], axis=1)
    return data


def run_experiment(exp_name):
    """Run a specific experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}\n")

    # Get configuration
    config = EXPERIMENTS[exp_name]
    print(f"Description: {config['description']}")

    # Setup
    setup_directories()
    data = load_data()

    # Create model
    model = NCA(**config["model"])
    print(f"\nModel created with {config['model']['N_CHANNELS']} channels")

    # Create trainer
    trainer = NCA_Trainer(
        NCA_model=model,
        data=data,
        DATA_AUGMENTER=DataAugmenter,
        model_filename=f"{exp_name}_{config['name']}",
        MODEL_DIRECTORY=PATHS["models"],
        LOG_DIRECTORY=PATHS["logs"]
    )

    # Setup optimizer
    train_config = config["train"]
    schedule = optax.exponential_decay(
        init_value=train_config["learning_rate"],
        transition_steps=train_config["iters"],
        decay_rate=0.99
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )

    # Train
    print(f"\nTraining for {train_config['iters']} iterations...")
    trainer.train(
        t=train_config["t"],
        iters=train_config["iters"],
        optimiser=optimiser,
        WARMUP=train_config["warmup"],
        wandb_args={
            "project": "my_project",
            "name": f"{exp_name}_{config['name']}",
            "group": "experiments",
            "config": config
        },
        LOG_EVERY=100
    )

    print(f"\n✓ Experiment {exp_name} complete!")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run"
    )
    args = parser.parse_args()

    if args.experiment == "all":
        # Run all experiments
        for exp_name in EXPERIMENTS.keys():
            run_experiment(exp_name)
    else:
        # Run specific experiment
        run_experiment(args.experiment)


if __name__ == "__main__":
    main()
```

### 3. Run Experiments

```bash
# Run specific experiment
python scripts/run_experiment.py exp1

# Run all experiments
python scripts/run_experiment.py all
```

---

## Version Control Best Practices

### Git Workflow

```bash
# Create feature branch for your project
git checkout -b experiments/my-project

# Track important files
git add Experiments/my_project/
git add Experiments/my_project/config/
git add Experiments/my_project/scripts/
git add Experiments/my_project/notebooks/

# Don't track large files
echo "*.eqx" >> .gitignore
echo "*.mp4" >> .gitignore
echo "logs/" >> .gitignore
echo "wandb/" >> .gitignore

# Commit regularly
git commit -m "Add experiment configuration for [project]"
git commit -m "Add baseline training script"
git commit -m "Results: baseline achieves X loss"
```

### What to Track

**DO track:**
- Configuration files
- Scripts
- Notebooks (clear output first)
- Small data files
- README and documentation
- Requirements files

**DON'T track:**
- Model checkpoints (*.eqx)
- Large datasets
- Video files
- Log directories
- Temporary files

### Using Git LFS for Large Files

If you need to track large files:

```bash
# Install Git LFS
git lfs install

# Track specific file types
git lfs track "*.eqx"
git lfs track "*.npy"
git lfs track "*.mp4"

# Commit the tracking file
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## Logging and Tracking

### Using Weights & Biases

1. **Install and login:**
```bash
pip install wandb
wandb login
```

2. **Configure in your script:**
```python
trainer.train(
    # ... other args ...
    wandb_args={
        "project": "my-phd-project",          # Project name
        "name": f"exp_{variant}_{timestamp}", # Run name
        "group": "baseline_experiments",       # Group related runs
        "tags": ["nca", "pattern-formation"], # Searchable tags
        "notes": "Testing effect of channel count",
        "config": {                            # Log hyperparameters
            "channels": N_CHANNELS,
            "kernels": KERNEL_STR,
            "learning_rate": 1e-4,
        }
    }
)
```

3. **Log custom metrics:**
```python
import wandb

wandb.log({
    "custom_metric": value,
    "pattern_quality": score,
})
```

### Tensorboard (Alternative)

The trainer automatically logs to Tensorboard:

```bash
# View logs
tensorboard --logdir=logs/

# Open browser to http://localhost:6006
```

### Organizing Runs

Use a consistent naming scheme:

```
project_variant_date
└─ micropatterns_baseline_2024-01-15
└─ micropatterns_highres_2024-01-16
└─ micropatterns_ablation_nokernels_2024-01-17
```

---

## Running on HPC Clusters

### Creating SLURM Scripts

Create `scripts/submit_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=nca_exp
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module load cuda/12.0
module load python/3.10

# Activate environment
source ~/venvs/jax/bin/activate

# Run experiment
cd $SLURM_SUBMIT_DIR
python scripts/run_experiment.py exp1

# Notify completion
echo "Job $SLURM_JOB_ID completed at $(date)"
```

### Submitting Jobs

```bash
# Submit single job
sbatch scripts/submit_job.sh

# Submit array of jobs
sbatch --array=1-10 scripts/submit_job.sh

# Check status
squeue -u $USER

# Cancel job
scancel <job_id>
```

### Multi-GPU Training

For data parallelism:

```python
import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Setup sharding
devices = jax.devices()
n_devices = len(devices)
sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,)))

# Use in trainer
trainer = NCA_Trainer(
    # ... other args ...
    SHARDING=sharding
)
```

---

## Reproducibility

### 1. Fix Random Seeds

```python
import jax.random as jr

# Set seed at start of script
SEED = 42
key = jr.PRNGKey(SEED)

# Use key explicitly
key, subkey = jr.split(key)
data = jr.normal(subkey, shape=(10,))
```

### 2. Log Environment

Create `scripts/log_environment.py`:

```python
import sys
import jax
import equinox
import optax

print("Environment Information")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")
print(f"Equinox: {equinox.__version__}")
print(f"Optax: {optax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Platform: {jax.default_backend()}")
```

Run and save:
```bash
python scripts/log_environment.py > results/environment.txt
```

### 3. Save Configurations

```python
import json
from datetime import datetime

# Save configuration with results
metadata = {
    "timestamp": datetime.now().isoformat(),
    "experiment": exp_name,
    "config": config,
    "seed": SEED,
    "git_commit": "abc123",  # Get with: git rev-parse HEAD
}

with open(f"results/{exp_name}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### 4. Archive Final Results

```bash
# Create archive of experiment
tar -czf my_project_2024-01-15.tar.gz \
    config/ \
    scripts/ \
    results/ \
    figures/ \
    README.md \
    environment.txt

# Upload to persistent storage
# (university server, AWS S3, etc.)
```

---

## Writing Papers

### Organizing Figures

Create `notebooks/generate_figures.ipynb`:

```python
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def save_figure(fig, name):
    """Save in multiple formats."""
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    print(f"Saved {name}")

# Generate Figure 1: Model architecture
fig, ax = plt.subplots(figsize=(6, 4))
# ... plotting code ...
save_figure(fig, "figure1_architecture")

# Generate Figure 2: Training results
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
# ... plotting code ...
save_figure(fig, "figure2_results")
```

### Results Table

Create `scripts/generate_results_table.py`:

```python
import pandas as pd

results = []
for exp_name in EXPERIMENTS:
    # Load results
    result = {
        "Experiment": exp_name,
        "Channels": config["model"]["N_CHANNELS"],
        "Final Loss": final_loss,
        "Training Time": training_time,
    }
    results.append(result)

df = pd.DataFrame(results)
print(df.to_latex(index=False))
df.to_csv("results/results_table.csv")
```

### Writing Methods Section

Template for describing your experiments:

```markdown
## Methods

### Model Architecture
We used Neural Cellular Automata (NCA) based on [citation].
The model consists of [N] channels per cell, with [kernels]
spatial kernels. Updates were computed using [activation]
activation and applied stochastically with fire rate [rate].

### Training
Models were trained for [N] iterations using [optimizer]
with learning rate [lr]. Training data consisted of [description].
We used [regularization] to [purpose].

### Evaluation
Models were evaluated on [metrics]. We measured [what]
by [how]. Statistical significance was assessed using [test].

### Implementation
All experiments were implemented in JAX [version] using
Equinox [version]. Training was performed on [hardware].
Code is available at [URL].
```

---

## Example: Complete Research Workflow

Let's walk through a complete example:

### Week 1: Setup
```bash
# Create project
mkdir Experiments/morphogen_gradients
cd Experiments/morphogen_gradients

# Set up structure
mkdir -p {config,scripts,notebooks,models,logs,results,figures}

# Create configs (as shown above)
# Create run script (as shown above)

# Version control
git checkout -b experiments/morphogen-gradients
git add config/ scripts/ README.md
git commit -m "Initial setup for morphogen gradient experiments"
```

### Week 2-4: Experiments
```bash
# Run baseline
python scripts/run_experiment.py baseline

# Analyze, adjust config
# Run variants
python scripts/run_experiment.py variant1
python scripts/run_experiment.py variant2

# Track in git
git add config/experiment_configs.py
git commit -m "Add variant configurations"
```

### Week 5-6: Analysis
```jupyter
# In notebooks/analysis.ipynb

# Load results
baseline_model = load_model("models/baseline.eqx")
variant1_model = load_model("models/variant1.eqx")

# Compare
compare_models([baseline_model, variant1_model])

# Generate figures
create_comparison_plots()
save_figure(fig, "figure3_comparison")
```

### Week 7: Writing
```bash
# Generate all figures
jupyter nbconvert --execute notebooks/generate_figures.ipynb

# Create results table
python scripts/generate_results_table.py

# Archive
tar -czf morphogen_gradients_final.tar.gz \
    config/ scripts/ results/ figures/

# Merge to main
git add results/ figures/
git commit -m "Final results for morphogen gradient experiments"
git push origin experiments/morphogen-gradients
```

---

## Checklist for a Complete Experiment

- [ ] Clear research question defined
- [ ] Configuration files created
- [ ] Scripts written and tested
- [ ] Baseline experiment run
- [ ] Results logged (wandb/tensorboard)
- [ ] Ablations/variants completed
- [ ] Analysis notebook created
- [ ] Figures generated in publication quality
- [ ] Results table created
- [ ] Code committed to version control
- [ ] Environment documented
- [ ] Results archived
- [ ] README updated with findings

---

## Tips for PhD Students

1. **Document as you go** - Write README updates immediately after experiments
2. **Version everything** - If you can't reproduce it, it didn't happen
3. **Start with simple baselines** - Get something working before optimizing
4. **Keep a research log** - Daily notes on what you tried and why
5. **Make figures early** - Create plotting code from day 1
6. **Automate repetitive tasks** - Write scripts for common operations
7. **Ask for code review** - Have others look at your experiment setup
8. **Plan for scale** - Design for HPC from the start
9. **Save intermediate results** - Don't rely on being able to re-run everything
10. **Celebrate small wins** - Research is hard, acknowledge progress!

---

Good luck with your research! 🚀
