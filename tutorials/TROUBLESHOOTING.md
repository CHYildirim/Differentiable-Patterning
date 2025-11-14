# Troubleshooting Guide

Common errors and their solutions.

## Installation Issues

### Problem: JAX not finding CUDA/GPU

**Error:**
```
RuntimeError: CUDA backend failed to initialize
```

**Solution:**
1. Check CUDA version matches JAX requirements (CUDA 12 for current version)
2. Verify GPU visibility:
   ```python
   import jax
   print(jax.devices())  # Should show GPU devices
   ```
3. Reinstall with correct CUDA version:
   ```bash
   pip install -r requirements_gpu.txt --force-reinstall
   ```
4. If persistent, use CPU version for development:
   ```bash
   pip install -r requirements_cpu.txt
   ```

### Problem: Import errors

**Error:**
```
ModuleNotFoundError: No module named 'NCA'
```

**Solution:**
Make sure you're running from the correct directory:
```python
import sys
sys.path.append('..')  # Add parent directory to path
```

Or install in development mode:
```bash
pip install -e .
```

---

## Memory Issues

### Problem: Out of memory (OOM) errors

**Error:**
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   BATCHES = 1  # Instead of 2 or 4
   ```

2. **Use gradient checkpointing:**
   ```python
   trainer.train(
       LOOP_AUTODIFF="checkpointed"  # Instead of "lax"
   )
   ```

3. **Reduce grid size:**
   ```python
   DOWNSAMPLE = 8  # Instead of 4
   SIZE = 64      # Instead of 128
   ```

4. **Reduce channels:**
   ```python
   model = NCA(N_CHANNELS=8)  # Instead of 16 or 20
   ```

5. **Clear cache between runs:**
   ```python
   jax.clear_caches()
   ```

### Problem: Memory leak during training

**Symptoms:** Memory usage grows over time

**Solution:**
```python
trainer.train(
    CLEAR_CACHE_EVERY=100  # Clear every 100 iterations
)
```

---

## Training Issues

### Problem: NaN values during training

**Error:**
```
Warning: NaN detected in loss
```

**Solutions:**

1. **Reduce learning rate:**
   ```python
   schedule = optax.exponential_decay(
       init_value=1e-5,  # Smaller learning rate
       transition_steps=TRAINING_STEPS,
       decay_rate=0.99
   )
   ```

2. **Check for numerical instability in data:**
   ```python
   print(f"Data range: {data.min()} to {data.max()}")
   data = jnp.clip(data, 0, 1)  # Ensure valid range
   ```

3. **Use gradient clipping:**
   ```python
   optimiser = optax.chain(
       optax.clip_by_global_norm(1.0),  # Add gradient clipping
       optax.adam(learning_rate)
   )
   ```

4. **For PDEs, reduce dt or dx:**
   ```python
   solver = PDE_solver(func, dt=0.1)  # Smaller timestep
   func = F(dx=0.25)  # Smaller spatial step
   ```

### Problem: Training not converging

**Symptoms:** Loss stays high or fluctuates

**Solutions:**

1. **Increase training steps:**
   ```python
   trainer.train(iters=5000)  # More iterations
   ```

2. **Adjust warmup period:**
   ```python
   trainer.train(WARMUP=50)  # Longer warmup
   ```

3. **Check data preprocessing:**
   ```python
   # Ensure data is in [0, 1] range
   data = (data - data.min()) / (data.max() - data.min())
   ```

4. **Try different optimizer:**
   ```python
   optimiser = optax.adamw(1e-3)  # Instead of nadam
   ```

5. **Increase model capacity:**
   ```python
   model = NCA(N_CHANNELS=32)  # More channels
   ```

---

## Data Loading Issues

### Problem: Image not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'crab.png'
```

**Solution:**
Check the path is correct relative to your current directory:
```python
# If running from tutorials/
data = load_emoji_sequence(
    ["crab.png"],
    impath_emojis="../demo/demo_data/"
)

# If running from demo/
data = load_emoji_sequence(
    ["crab.png"],
    impath_emojis="demo_data/"
)
```

### Problem: Data shape mismatch

**Error:**
```
ValueError: Incompatible shapes for broadcasting
```

**Solution:**
Check data dimensions match expectations:
```python
print(f"Data shape: {data.shape}")
# Should be: (Batch, Time, Channels, Height, Width)

# Fix if needed:
data = data[None, ...]  # Add batch dimension
data = jnp.expand_dims(data, 0)  # Alternative
```

---

## JAX-Specific Issues

### Problem: "Tracer" errors

**Error:**
```
UnexpectedTracerError: Encountered an unexpected tracer
```

**Cause:** Mixing Python control flow with JAX arrays

**Solution:**
Use JAX control flow:
```python
# DON'T do this:
if x > 0:
    y = x + 1

# DO this instead:
y = jax.lax.cond(x > 0, lambda: x + 1, lambda: x)

# Or for arrays:
y = jnp.where(x > 0, x + 1, x)
```

### Problem: Random numbers not working

**Error:**
```
TypeError: JAX random requires a key
```

**Solution:**
Always use JAX random with explicit keys:
```python
import jax.random as jr

# Create key
key = jr.PRNGKey(42)

# Split key for multiple operations
key, subkey = jr.split(key)
x = jr.normal(subkey, shape=(10,))

key, subkey = jr.split(key)
y = jr.uniform(subkey, shape=(10,))
```

### Problem: Slow compilation

**Symptoms:** First run is very slow

**This is normal!** JAX JIT-compiles functions on first call.

**Tips:**
1. Use smaller problem sizes during development
2. The second run will be much faster
3. Use `--xla_dump_to=/tmp/xla` to debug compilation issues

---

## Model Loading/Saving Issues

### Problem: Can't load saved model

**Error:**
```
ValueError: Parameter mismatch when loading model
```

**Solution:**
Ensure model architecture matches:
```python
# When saving, note the configuration:
model = NCA(N_CHANNELS=20, KERNEL_STR=["ID", "GRAD", "LAP"])
model.save("my_model.eqx")

# When loading, use EXACT same configuration:
model = NCA(N_CHANNELS=20, KERNEL_STR=["ID", "GRAD", "LAP"])
model = model.load("my_model.eqx")
```

---

## Visualization Issues

### Problem: Plot shows wrong dimensions

**Error:**
```
ValueError: could not broadcast input array
```

**Solution:**
Use `einops.rearrange` to fix dimensions:
```python
from einops import rearrange

# For NCA output: (Time, Channels, Height, Width)
# To show RGB over time:
rgb = rearrange(trajectory[:, :3], "T C H W -> H (T W) C")
plt.imshow(rgb)

# To show single frame:
frame = rearrange(trajectory[0, :3], "C H W -> H W C")
plt.imshow(frame)
```

### Problem: Colors look wrong

**Solution:**
1. Ensure RGB values in [0, 1]:
   ```python
   rgb = jnp.clip(rgb, 0, 1)
   ```

2. For single channel, use colormap:
   ```python
   plt.imshow(data[0], cmap='viridis')
   ```

---

## Performance Issues

### Problem: Training is too slow

**Solutions:**

1. **Use GPU:**
   ```python
   print(jax.devices())  # Check GPU is available
   ```

2. **Reduce problem size:**
   - Smaller grid: `SIZE = 64`
   - Fewer channels: `N_CHANNELS = 8`
   - Fewer steps: `t = 32`

3. **Use JIT compilation:**
   ```python
   @jax.jit
   def run_model(x0):
       return model.run(64, x0)
   ```

4. **Batch operations:**
   ```python
   # Instead of for-loop, use vmap:
   run_batch = jax.vmap(model.run, in_axes=(None, 0))
   trajectories = run_batch(64, batch_of_x0)
   ```

### Problem: Too much disk space used

**Cause:** Tensorboard logs and model checkpoints

**Solution:**
```bash
# Clean old logs
rm -rf logs/*

# Clean old models
rm -rf models/*.eqx
```

---

## Debugging Tips

### Enable better error messages

```python
# More informative errors
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# Disable JIT for easier debugging
jax.config.update('jax_disable_jit', True)
```

### Check intermediate values

```python
# Add print debugging
def debug_fn(x):
    jax.debug.print("x shape: {}, min: {}, max: {}",
                    x.shape, x.min(), x.max())
    return x

x = debug_fn(x)
```

### Profile performance

```python
import time

start = time.time()
result = model.run(100, x0)
print(f"Time: {time.time() - start:.2f}s")
```

---

## Still Stuck?

1. **Check existing issues:** Search the codebase's issues on GitHub
2. **Look at working examples:** The `demo/` and `tests/` folders have working code
3. **Simplify:** Create a minimal example that reproduces the problem
4. **Read the docs:**
   - [JAX documentation](https://jax.readthedocs.io/)
   - [Equinox documentation](https://docs.kidger.site/equinox/)
5. **Ask for help:** Create an issue with:
   - Full error message
   - Minimal code to reproduce
   - System information (GPU, JAX version, etc.)

---

## Preventive Best Practices

1. **Always check data shapes:**
   ```python
   print(f"Shape: {data.shape}, dtype: {data.dtype}")
   ```

2. **Use assertions:**
   ```python
   assert data.shape[1] == 2, f"Expected 2 timesteps, got {data.shape[1]}"
   ```

3. **Start small:**
   - Small grid (32x32)
   - Few channels (8)
   - Few iterations (100)
   Then scale up.

4. **Save frequently:**
   ```python
   model.save(f"checkpoint_{iteration}.eqx")
   ```

5. **Monitor training:**
   - Use wandb logging
   - Check loss curves
   - Visualize intermediate results
