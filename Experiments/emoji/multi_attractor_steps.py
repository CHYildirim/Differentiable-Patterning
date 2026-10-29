"""Evaluation of multi-attractor emoji NCAs: which target does a seed grow into?

A multi-attractor model (``data.emoji.task: multi_attractor``) is trained to
grow a different emoji from each seed in ``data.emoji.pairs``. The helpers here
rebuild those seeds and targets from a bundle's saved config, write other
colours into the seed, roll many seeds out at once, and label each final state
with the target it is closest to.

Distances are the mean squared error of the observed channels (RGBA) over the
whole padded grid, as in the training loss. A final state counts as a target
when its distance is below ``tolerance`` times the smallest distance between
two different targets, so the threshold does not depend on the emoji used.

``rollout(..., dynamics=True)`` also classifies the state after every step as
``spatial_position_scan_combined.py`` does (``scan_class``), and
``attractor_dynamics`` turns that into its stable-attractor measures
(``reaches_pure_attractor``, ``steps_to_pure_attractor``,
``converged_before_end``).

This is ``multi_attractor.py`` plus what ``multi_attractor_steps_explorer.py``
needs from the scan and plot scripts:

- ``rollout_until_stable``: the dynamic system of
  ``spatial_position_scan_dynamic.py`` (every run until its image is stable,
  ``steps_to_stable_image``)
- seeds at the 9 positions of the scans (``scan_positions``, ``pixel_states``,
  ``shifts`` in the rollouts)
- the colour sweeps of the scans (``colour_sweeps``), with more alphas where the
  outcome changes (``adaptive_sweeps``, ``boundary_windows``), for the plots of
  ``plot_steps_vs_spectrum_en.py`` and ``plot_steps_vs_spectrum_dynamic.py``
- a PCA of the full states along the runs (``rollout_states``, ``state_pca``,
  ``pca_project``)
"""

from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from Experiments.emoji.config_helpers import _pad_tuple, build_hidden_seeds, load_data
from NCA.model.factory import PORTABLE_MODEL_FAMILIES, build_model

# Labels of final states that are not one of the targets
OTHER, DIED, DIVERGED = -1, -2, -3
OUTCOME_NAMES = {OTHER: "other", DIED: "died", DIVERGED: "diverged"}
# A cell is alive when its alpha is above this, as in the usual NCA convention
ALIVE_ALPHA = 0.1

# Stable-attractor criterion of spatial_position_scan_combined.py (same names and values).
# Class of a state: the target with the smallest MSE, INTERMEDIATE when at least two
# targets each own MIXED_MIN_FRACTION of the active pixels, NO_VISIBLE_PATTERN when
# no pixel is active.
ACTIVE_ALPHA_THRESHOLD = 0.10
MIXED_MIN_FRACTION = 0.20
# A run has settled when the smoothed RGBA change per step ends at most
# CONVERGENCE_SETTLED_FRACTION of its peak; it is stable from the last step after
# which the change stays below max(ABS_TOL, REL_TOL * residual change).
CONVERGENCE_ABS_TOL = 1e-5
CONVERGENCE_REL_TOL = 3.0
CONVERGENCE_TAIL_STEPS = 16
CONVERGENCE_SETTLED_FRACTION = 0.20
CONVERGENCE_SMOOTH_WINDOW = 8


def is_multi_attractor(cfg):
    return cfg.data.dataset == "emojis" and cfg.data.emoji.task == "multi_attractor"


def emoji_name(filename):
    return Path(filename).stem


def seed_description(initial):
    """Short text for a pair's ``initial`` entry, e.g. ``patch 1`` or ``pixel [3, 4]``."""
    if isinstance(initial, str):
        return "full"
    mode = initial.get("mode", "full")
    if mode == "patch":
        return f"patch {initial.get('size', 12)}"
    if mode == "pixel":
        return f"pixel {list(initial.get('channels', [initial.get('channel', 0)]))}"
    return mode


def target_names(cfg):
    """Distinct target emoji names, in the order of ``data.emoji.pairs``."""
    return list(dict.fromkeys(emoji_name(pair.target) for pair in cfg.data.emoji.pairs))


def summary_row(bundle):
    """One table row of the settings that differ between multi-attractor runs."""
    cfg = bundle.config
    emoji = cfg.data.emoji
    names = target_names(cfg)
    training = bundle.manifest.get("training", {})
    return {
        "model_id": bundle.id,
        "created_at": str(bundle.manifest.created_at),
        "collection": str(bundle.manifest.collection),
        "experiment": bundle.manifest.get("experiment"),
        "wandb_group": cfg.logging.wandb.group,
        "attractors": len(names),
        "targets": " + ".join(names),
        "seed": " / ".join(dict.fromkeys(seed_description(pair.initial) for pair in emoji.pairs)),
        "family": cfg.model.family,
        "channels": cfg.model.channels,
        "padding": cfg.model.padding,
        "fire_rate": cfg.model.fire_rate,
        "parameter_noise": cfg.model.parameter_noise_level,
        "t": cfg.run.t,
        "iterations": cfg.run.iterations,
        "batches": cfg.data.batches,
        "stability_mode": cfg.experiment.stability_mode,
        "terminal_carry": emoji.terminal_carry.enabled,
        "regeneration": emoji.regeneration.enabled,
        "repeat": cfg.run.repeat,
        "best_loss": training.get("best_loss"),
        "best_iteration": training.get("best_iteration"),
    }


def load_model(bundle, parameter_noise=True):
    """The bundle's model; ``parameter_noise=False`` switches off the weight noise
    used in training (a static option, so the saved weights load unchanged)."""
    if parameter_noise:
        return bundle.load_model()
    bundle.verify()
    model_config = replace(
        bundle.config.model,
        family=PORTABLE_MODEL_FAMILIES.get(bundle.config.model.family, bundle.config.model.family),
        parameter_noise_level=0.0,
    )
    model, _ = build_model(model_config, key=jax.random.PRNGKey(0))
    return model.load(bundle.checkpoint_path)


def load_seeds_and_targets(cfg, model_channels, impath):
    """Seeds and targets of a multi-attractor config, as training builds them.

    Returns ``(seeds, targets, pair_target, data)``: ``seeds`` [pair, model
    channels, H, W] and ``targets`` [target, data channels, H, W] are padded
    like the training pool; ``pair_target`` gives each pair's index into
    ``targets`` (pairs may share a target); ``data`` is the unpadded
    [pair, time, data channels, H, W] array, for ``verify_evaluation_input``.
    """
    data, _ = load_data(cfg.data, impath=impath)
    data = np.asarray(data, dtype=np.float32)
    pairs, _, data_channels, height, width = data.shape
    hidden = build_hidden_seeds(cfg.data, data, model_channels)
    if hidden is None:
        hidden = np.zeros((pairs, model_channels - data_channels, height, width), np.float32)
    seeds = np.concatenate([data[:, 0], hidden], axis=1)

    names = target_names(cfg)
    pair_target = np.array([names.index(emoji_name(pair.target)) for pair in cfg.data.emoji.pairs])
    first_pair = [list(pair_target).index(index) for index in range(len(names))]
    targets = data[first_pair, -1]

    pad = _pad_tuple(cfg.data.emoji.pad)
    if pad is not None:
        top, bottom, left, right = pad
        widths = ((0, 0), (0, 0), (top, bottom), (left, right))
        seeds, targets = np.pad(seeds, widths), np.pad(targets, widths)
    return seeds, targets, pair_target, data


def seed_footprint(seeds):
    """[H, W] mask of the pixels any training seed sets: one pixel for pixel and
    1x1 patch seeds, a square for larger patches."""
    footprint = np.any(seeds != 0, axis=(0, 1))
    if not footprint.any():
        footprint[seeds.shape[-2] // 2, seeds.shape[-1] // 2] = True
    return footprint


def seed_colours(seeds, footprint, data_channels=4):
    """Mean RGBA of each training seed over the footprint, [pair, 4]."""
    return seeds[:, :data_channels, footprint].mean(axis=-1)


def hsva_to_rgba(hsva):
    """[..., 4] hue, saturation, value, alpha (all in [0, 1]) to RGBA."""
    hsva = np.asarray(hsva, dtype=np.float32)
    rgb = hsv_to_rgb(np.clip(hsva[..., :3], 0.0, 1.0))
    return np.concatenate([rgb, hsva[..., 3:4]], axis=-1).astype(np.float32)


def rgba_to_hsva(rgba):
    rgba = np.asarray(rgba, dtype=np.float32)
    hsv = rgb_to_hsv(np.clip(rgba[..., :3], 0.0, 1.0))
    return np.concatenate([hsv, rgba[..., 3:4]], axis=-1)


def coloured_seeds(base, footprint, rgba):
    """Copies of the ``base`` seed [C, H, W] with the footprint set to each colour in
    ``rgba`` [N, 4]; the hidden channels keep the base seed's values."""
    rgba = np.asarray(rgba, dtype=np.float32)
    states = np.repeat(np.asarray(base, dtype=np.float32)[None], len(rgba), axis=0)
    states[:, : rgba.shape[1], footprint] = rgba[:, :, None]
    return states


def scan_class(visible, targets):
    """Class of a state at the training position, as ``classify_state_jax`` in
    spatial_position_scan_combined.py: 0 .. n-1 the target, n INTERMEDIATE,
    n + 1 NO_VISIBLE_PATTERN. ``visible`` [data channels, H, W], ``targets``
    [n, data channels, H, W]."""
    n = targets.shape[0]
    mse = jnp.mean((visible[None] - targets) ** 2, axis=(1, 2, 3))
    per_pixel_error = jnp.mean((visible[None, :3] - targets[:, :3]) ** 2, axis=1)
    if visible.shape[0] >= 4:
        active = visible[3] > ACTIVE_ALPHA_THRESHOLD
    else:
        active = jnp.max(jnp.abs(visible[:3]), axis=0) > 0.10
    active_pixels = jnp.sum(active)
    if n == 1:
        mixed = False
    else:
        # A pixel counts for a target only when that target is strictly the closest
        ordered = jnp.sort(per_pixel_error, axis=0)
        counted = (ordered[0] < ordered[1]) & active
        nearest = jnp.argmin(per_pixel_error, axis=0)
        counts = jnp.stack([jnp.sum(counted & (nearest == k)) for k in range(n)])
        # fraction >= MIXED_MIN_FRACTION in integers, so a tie (exactly 20 %) is decided as by
        # the scan's final numpy classification, not by float32 rounding
        mixed = jnp.sum(counts * 1000 >= round(MIXED_MIN_FRACTION * 1000) * active_pixels) >= 2
    label = jnp.where(mixed, n, jnp.argmin(mse))
    return jnp.where(active_pixels == 0, n + 1, label).astype(jnp.int32)


def mixed_states(base, targets, weights):
    """Start states between the targets: copies of ``base`` [C, H, W] whose RGBA is
    the weighted sum of the target images (``weights`` [N, target], e.g. half crab
    and half microbe), clipped to [0, 1]; the hidden channels keep the base values."""
    targets = np.asarray(targets, dtype=np.float32)
    rgba = np.clip(np.tensordot(np.asarray(weights, dtype=np.float32), targets, axes=1), 0.0, 1.0)
    states = np.repeat(np.asarray(base, dtype=np.float32)[None], len(rgba), axis=0)
    states[:, : targets.shape[1]] = rgba
    return states


def align(visible, shift, circular):
    """``shift_for_analysis_jax`` of the scans: move a state [C, H, W] by ``shift``
    (dy, dx) so a seed placed elsewhere sits at the training position. With
    ``circular`` padding it wraps around, otherwise what is pushed out is lost."""
    rolled = jnp.roll(visible, shift=(shift[0], shift[1]), axis=(-2, -1))
    if circular:
        return rolled
    rows = jnp.arange(visible.shape[-2]) - shift[0]
    cols = jnp.arange(visible.shape[-1]) - shift[1]
    keep = ((rows >= 0) & (rows < visible.shape[-2]))[:, None] & ((cols >= 0) & (cols < visible.shape[-1]))[None, :]
    return rolled * keep


def _rollout(model, state, key, targets, steps, stride, shift, circular):
    """One trajectory of ``steps`` updates: RGBA every ``stride`` steps and the
    distance to each target at every step, both including the seed; and after
    every step the scan class and the RGBA change since the step before. Distances
    and classes are taken after moving the state by ``shift`` (see ``align``)."""
    data_channels = targets.shape[1]

    def distance(visible):
        return jnp.mean((align(visible, shift, circular)[None] - targets) ** 2, axis=(1, 2, 3))

    def step(state, index):
        new = model(state, lambda x: x, key=jax.random.fold_in(key, index))
        visible = new[:data_channels]
        change = jnp.mean((visible - state[:data_channels]) ** 2)
        return new, (distance(visible), scan_class(align(visible, shift, circular), targets), change)

    def frame(state, start):
        state, outputs = jax.lax.scan(step, state, start + jnp.arange(stride))
        return state, (state[:data_channels], outputs)

    _, (frames, (distances, classes, changes)) = jax.lax.scan(frame, state, stride * jnp.arange(steps // stride))
    frames = jnp.concatenate([state[None, :data_channels], frames])
    distances = jnp.concatenate([distance(state[:data_channels])[None], distances.reshape(-1, len(targets))])
    return frames, distances, classes.reshape(-1), changes.reshape(-1)


@eqx.filter_jit
def _rollout_batch(model, states, keys, targets, steps, stride, shifts, circular):
    return jax.vmap(lambda state, key, shift: _rollout(model, state, key, targets, steps, stride, shift, circular))(
        states, keys, shifts)


def chunk_size(model, height, width, memory_gb):
    """How many seeds can be rolled out at once in about ``memory_gb`` of device memory.

    The largest arrays are the hidden layer's feature maps (``N_FEATURES`` x H x W
    floats per seed). A compiled rollout holds about three of them per seed;
    one more is allowed for the GPU convolution workspace.
    """
    features = getattr(model, "N_FEATURES", 4 * model.N_CHANNELS)
    per_seed = 4 * 4 * features * height * width
    return max(1, int(memory_gb * 2**30 // per_seed))


def _padded_batches(states, key, shifts, model, memory_gb, shared_key=False):
    """Seeds, per-run keys and shifts padded to whole chunks of ``chunk_size``, so
    every chunk has the same shape and compiles once; and the chunk length. With
    ``shared_key`` every run uses ``key`` itself (as in the scans, where step t of
    every run uses the same key), so a run does not depend on its place in the batch."""
    states = np.asarray(states, dtype=np.float32)
    count = len(states)
    chunk = min(count, chunk_size(model, *states.shape[-2:], memory_gb))
    keys = jnp.repeat(key[None], count, axis=0) if shared_key else jax.random.split(key, count)
    shifts = np.zeros((count, 2), np.int32) if shifts is None else np.asarray(shifts, np.int32)
    padding = -count % chunk
    states = np.concatenate([states, np.zeros((padding, *states.shape[1:]), np.float32)])
    keys = jnp.concatenate([keys, jnp.repeat(keys[:1], padding, axis=0)])
    shifts = np.concatenate([shifts, np.zeros((padding, 2), np.int32)])
    return states, keys, shifts, chunk


def rollout(model, states, targets, steps, key, frames=1, memory_gb=1.0, dynamics=False, shifts=None,
            circular=False, shared_key=False):
    """Roll out each seed in ``states`` [N, C, H, W] for ``steps`` updates.

    Returns the distance to each target at every step, [N, steps + 1, target],
    and the RGBA every ``steps / frames`` steps, [N, frames + 1, 4, H, W]
    (``frames=1``: the seed and the final state). With ``dynamics`` also the
    scan class (``scan_class``) and the RGBA change after every step, both
    [N, steps], for ``attractor_dynamics``. Only these are kept, and the
    seeds are run in chunks that fit in about ``memory_gb`` (see ``chunk_size``).

    ``shifts`` [N, 2] (dy, dx) move each state to the training position before
    distances and classes are taken (seeds placed elsewhere, see
    ``scan_positions``); ``circular``: the model pads circularly. ``shared_key``:
    every run uses the same random keys (see ``_padded_batches``).
    """
    if steps % frames:
        raise ValueError(f"steps ({steps}) must be a multiple of frames ({frames})")
    count = len(states)
    states, keys, shifts, chunk = _padded_batches(states, key, shifts, model, memory_gb, shared_key)
    targets = jnp.asarray(targets)
    # rgba, distances, classes, changes
    outputs = ([], [], [], [])
    for start in range(0, len(states), chunk):
        results = _rollout_batch(
            model, jnp.asarray(states[start : start + chunk]), keys[start : start + chunk],
            targets, int(steps), int(steps // frames), jnp.asarray(shifts[start : start + chunk]), bool(circular),
        )
        for output, result in zip(outputs, results):
            output.append(np.asarray(result))
    rgba, distances, classes, changes = (np.concatenate(output)[:count] for output in outputs)
    if dynamics:
        return distances, rgba, classes, changes
    return distances, rgba


# ----------------------------------------------------------------------
# Dynamic system (spatial_position_scan_dynamic.py): every run goes on until its
# visible image is stable. The image (RGBA clipped to [0, 1]) is stable from step N
# when it stays within STABLE_IMAGE_TOL (mean squared difference) of the image at
# step N for STABLE_WINDOW steps; then steps_to_stable_image = N and the run stops
# at N + STABLE_WINDOW. At least MIN_STEPS steps, at most MAX_STEPS (then the
# stable image is not reached). Same names and values as the scan.
STABLE_IMAGE_TOL = 1e-4
STABLE_WINDOW = 64
MIN_STEPS = 128
MAX_STEPS = 1024
CHUNK_STEPS = 32


@eqx.filter_jit
def _until_stable_chunk(model, slots, keys, chunk_steps, min_steps, max_steps, window, tolerance):
    """``chunk_steps`` more steps of every run that is not done yet (``advance_slots``
    of the dynamic scan). Step t + 1 of a run uses the same key as in ``rollout``."""
    data_channels = slots["anchor"].shape[1]

    def step(slots, _):
        state, t, done = slots["state"], slots["t"], slots["done"]
        new = jax.vmap(lambda s, k, i: model(s, lambda x: x, key=jax.random.fold_in(k, i)))(state, keys, t)
        image = jnp.clip(new[:, :data_channels], 0.0, 1.0)
        t_next = t + 1
        drift = jnp.mean((image - slots["anchor"]) ** 2, axis=(1, 2, 3))
        within = drift <= tolerance
        count = jnp.where(within, slots["count"] + 1, 0)
        anchor = jnp.where(within[:, None, None, None], slots["anchor"], image)
        anchor_step = jnp.where(within, slots["anchor_step"], t_next)
        stop = ((t_next >= min_steps) & (count >= window)) | (t_next >= max_steps)
        keep, keep4 = done, done[:, None, None, None]
        return {
            "state": jnp.where(keep4, state, new),
            "anchor": jnp.where(keep4, slots["anchor"], anchor),
            "t": jnp.where(keep, t, t_next),
            "anchor_step": jnp.where(keep, slots["anchor_step"], anchor_step),
            "count": jnp.where(keep, slots["count"], count),
            "drift": jnp.where(keep, slots["drift"], drift),
            "done": done | stop,
        }, None

    slots, _ = jax.lax.scan(step, slots, None, length=chunk_steps)
    return slots


@eqx.filter_jit
def _final_classes(states, shifts, targets, circular):
    data_channels = targets.shape[1]
    return jax.vmap(lambda s, shift: scan_class(align(s[:data_channels], shift, circular), targets))(states, shifts)


def rollout_until_stable(model, states, targets, key, memory_gb=1.0, shifts=None, circular=False,
                         min_steps=MIN_STEPS, max_steps=MAX_STEPS, window=STABLE_WINDOW,
                         tolerance=STABLE_IMAGE_TOL, chunk_steps=CHUNK_STEPS, shared_key=False):
    """The dynamic system: roll out each seed in ``states`` [N, C, H, W] until its
    visible image is stable (see STABLE_IMAGE_TOL). The first steps use the same
    keys as ``rollout`` (with the same ``shared_key``), so they are identical to a
    fixed-length rollout.

    Returns [N] arrays: ``steps_to_stable_image`` (-1 when not reached),
    ``stable_image_reached``, ``steps_run``, ``stable_image_drift`` and
    ``final_class`` (``scan_class`` of the final state, after ``shifts``), and the
    final RGBA ``final`` [N, 4, H, W].
    """
    count = len(states)
    states, keys, shifts, chunk = _padded_batches(states, key, shifts, model, memory_gb, shared_key)
    targets = jnp.asarray(targets)
    data_channels = targets.shape[1]
    outputs = {name: [] for name in ("anchor_step", "count", "t", "drift", "final_class", "final")}
    for start in range(0, len(states), chunk):
        block = jnp.asarray(states[start : start + chunk])
        slots = {
            "state": block,
            "anchor": jnp.clip(block[:, :data_channels], 0.0, 1.0),
            "t": jnp.zeros(len(block), jnp.int32),
            "anchor_step": jnp.zeros(len(block), jnp.int32),
            "count": jnp.zeros(len(block), jnp.int32),
            "drift": jnp.zeros(len(block), jnp.float32),
            "done": jnp.zeros(len(block), bool),
        }
        steps = 0
        while steps < max_steps and not bool(jnp.all(slots["done"])):
            slots = _until_stable_chunk(model, slots, keys[start : start + chunk], int(chunk_steps), int(min_steps),
                                        int(max_steps), int(window), float(tolerance))
            steps += chunk_steps
        block_shifts = jnp.asarray(shifts[start : start + chunk])
        for name in ("anchor_step", "count", "t", "drift"):
            outputs[name].append(np.asarray(slots[name]))
        outputs["final_class"].append(np.asarray(_final_classes(slots["state"], block_shifts, targets, bool(circular))))
        outputs["final"].append(np.asarray(slots["state"][:, :data_channels]))
    out = {name: np.concatenate(values)[:count] for name, values in outputs.items()}
    reached = out["count"] >= window
    return {
        "steps_to_stable_image": np.where(reached, out["anchor_step"], -1),
        "stable_image_reached": reached,
        "steps_run": out["t"],
        "stable_image_drift": out["drift"],
        "final_class": out["final_class"],
        "final": out["final"],
    }


# ----------------------------------------------------------------------
# PCA of the states: which directions in the full state (all channels and pixels) the runs
# move along. The states are kept every few steps; the PCA is computed from the Gram
# matrix of the fitted states, so it needs memory for samples x samples, not features^2.

@eqx.filter_jit
def _states_batch(model, states, keys, steps, stride):
    """The full state of every run every ``stride`` steps, including the seed. Step t uses the
    same key as in ``rollout``, so the runs are the same as there."""
    def single(state, key):
        def step(state, index):
            return model(state, lambda x: x, key=jax.random.fold_in(key, index)), None

        def frame(state, start):
            state, _ = jax.lax.scan(step, state, start + jnp.arange(stride))
            return state, state

        _, frames = jax.lax.scan(frame, state, stride * jnp.arange(steps // stride))
        return jnp.concatenate([state[None], frames])

    return jax.vmap(single)(states, keys)


def rollout_states(model, states, steps, key, every=4, memory_gb=1.0, shared_key=False, dtype=np.float16):
    """Roll out each seed in ``states`` [N, C, H, W] for ``steps`` updates and keep its full
    state (all channels) every ``every`` steps: [N, steps // every + 1, C, H, W] in ``dtype``
    (float16 halves the memory). With the same ``key`` and ``shared_key`` and the same batch
    of seeds, the runs are the same as in ``rollout``."""
    if steps % every:
        raise ValueError(f"steps ({steps}) must be a multiple of every ({every})")
    count = len(states)
    states, keys, _, chunk = _padded_batches(states, key, None, model, memory_gb, shared_key)
    # The kept states also take device memory: at most about memory_gb / 2 of them per chunk
    per_seed = 4 * (steps // every + 1) * int(np.prod(states.shape[1:]))
    chunk = max(1, min(chunk, int(memory_gb * 2**29 // per_seed)))
    padding = -count % chunk
    states = np.concatenate([states[:count], np.zeros((padding, *states.shape[1:]), np.float32)])
    keys = jnp.concatenate([keys[:count], jnp.repeat(keys[:1], padding, axis=0)])
    frames = []
    for start in range(0, len(states), chunk):
        block = _states_batch(model, jnp.asarray(states[start : start + chunk]), keys[start : start + chunk],
                              int(steps), int(every))
        frames.append(np.asarray(block).astype(dtype))
    return np.concatenate(frames)[:count]


def state_pca(states, fit, channels=None, standardize=True, n_components=10, block=256):
    """PCA of states [n, C, H, W] (any float dtype), fitted on the states where ``fit`` [n] is
    True and applied to all of them.

    ``channels``: the channels used (None: all). ``standardize``: every channel is divided by
    its standard deviation over the fitted states and pixels, so channels with large values
    (often the hidden ones) do not dominate; within a channel the pixels keep their relative
    scale. Returns a dict with ``scores`` [n, k] (the coordinates of every state), ``explained``
    [k] (fraction of the fitted variance), ``components`` [k, C', H, W] (unit vectors in the
    standardised space), ``mean`` [C', H, W], ``scale`` [C'], ``channels`` and ``n_fit``.
    """
    channels = list(range(states.shape[1])) if channels is None else list(channels)
    rows = np.flatnonzero(np.asarray(fit, bool))
    if len(rows) < 2:
        raise ValueError("the PCA needs at least two fitted states")
    shape = (len(channels), *states.shape[2:])
    features = int(np.prod(shape))

    def take(index):
        return np.asarray(states[index][:, channels], dtype=np.float64).reshape(len(index), -1)

    blocks = [rows[a : a + block] for a in range(0, len(rows), block)]
    mean = sum(take(index).sum(axis=0) for index in blocks) / len(rows)
    if standardize:
        pixels = features // len(channels)
        squares = sum(((take(index) - mean) ** 2).reshape(len(index), len(channels), pixels).sum(axis=(0, 2))
                      for index in blocks)
        scale = np.sqrt(squares / (len(rows) * pixels))
        scale = np.where(scale > 1e-12, scale, 1.0)
    else:
        scale = np.ones(len(channels))
    weights = np.repeat(1.0 / scale, features // len(channels))

    def centred(index):
        return (take(index) - mean) * weights

    # Gram matrix of the centred fitted states, block by block
    gram = np.zeros((len(rows), len(rows)))
    starts = np.cumsum([0] + [len(index) for index in blocks])
    for a, index_a in enumerate(blocks):
        data_a = centred(index_a)
        for b in range(a, len(blocks)):
            data_b = data_a if a == b else centred(blocks[b])
            product = data_a @ data_b.T
            gram[starts[a]:starts[a + 1], starts[b]:starts[b + 1]] = product
            gram[starts[b]:starts[b + 1], starts[a]:starts[a + 1]] = product.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][: min(n_components, len(rows))]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    singular = np.sqrt(values)
    keep = singular > 1e-12 * max(singular.max(), 1e-300)
    values, vectors, singular = values[keep], vectors[:, keep], singular[keep]
    # Components V = X^T U / s, block by block
    components = np.zeros((features, len(values)))
    for a, index in enumerate(blocks):
        components += centred(index).T @ (vectors[starts[a]:starts[a + 1]] / singular)
    # A fixed sign: the largest loading of every component is positive
    signs = np.sign(components[np.abs(components).argmax(axis=0), np.arange(components.shape[1])])
    components *= np.where(signs == 0, 1.0, signs)
    scores = np.concatenate([centred(np.arange(a, min(a + block, len(states)))) @ components
                             for a in range(0, len(states), block)])
    return {
        "scores": scores,
        "explained": values / max(np.trace(gram), 1e-300),
        "components": components.T.reshape(len(values), *shape),
        "mean": mean.reshape(shape),
        "scale": scale,
        "weights": weights,
        "channels": channels,
        "n_fit": len(rows),
    }


def pca_project(pca, states):
    """Coordinates of further states [m, C'', H, W] in a ``state_pca``: the PCA's channels are
    taken from them, channels they lack (e.g. the hidden channels of a target image) count as 0."""
    states = np.asarray(states, dtype=np.float64)
    full = np.zeros((len(states), len(pca["channels"]), *states.shape[2:]))
    for k, channel in enumerate(pca["channels"]):
        if channel < states.shape[1]:
            full[:, k] = states[:, channel]
    centred = (full.reshape(len(states), -1) - pca["mean"].ravel()) * pca["weights"]
    return centred @ pca["components"].reshape(len(pca["components"]), -1).T


# ----------------------------------------------------------------------
# Colour sweeps of the scans: a single seed pixel whose RGBA goes from one colour to
# another (alpha 0 .. 1), here with more alphas where the outcome changes.

def scan_positions(footprint):
    """The 9 seed positions of the scans (``all_positions``), centre first:
    [(name, y, x)]. The centre is the training position, the middle pixel of the
    training seeds' footprint; the others are the corners and edge midpoints."""
    height, width = footprint.shape
    ys, xs = np.nonzero(footprint)
    y, x = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    return [
        ("center", y, x), ("top_left", 0, 0), ("top_center", 0, x), ("top_right", 0, width - 1),
        ("center_left", y, 0), ("center_right", y, width - 1), ("bottom_left", height - 1, 0),
        ("bottom_center", height - 1, x), ("bottom_right", height - 1, width - 1),
    ]


def colour_sweeps(seeds, footprint, pair_target, names):
    """The colour sweeps of the scans (``build_scan_list``) for every pair of
    targets a, b (config order): ``normal`` from a's seed pixel to b's, and
    ``complementary`` between their complementary colours (1 - RGB). The seed
    pixel of a target is the middle pixel of its first training seed, RGBA only.
    Returns [dict(name, kind, a, b, start, end)]."""
    _, y, x = scan_positions(footprint)[0]
    first = [list(pair_target).index(k) for k in range(len(names))]
    pixel = [np.asarray(seeds[first[k], :4, y, x], np.float32) for k in range(len(names))]
    sweeps = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            tag = f"{names[a]} → {names[b]}"
            complement = [np.concatenate([1.0 - pixel[k][:3], pixel[k][3:]]) for k in (a, b)]
            sweeps.append({"name": f"seed colours: {tag}", "kind": "normal", "a": a, "b": b,
                           "start": pixel[a], "end": pixel[b]})
            sweeps.append({"name": f"complementary colours: {tag}", "kind": "complementary", "a": a, "b": b,
                           "start": complement[0], "end": complement[1]})
    return sweeps


def pixel_states(base, pixels, positions, training):
    """One state per seed pixel: empty except at ``positions`` [N, 2] (y, x), where
    the RGBA is ``pixels`` [N, 4] and the hidden channels are those of ``base``
    [C, H, W] at the ``training`` position (y, x). Also returns the shifts that
    move each seed to the training position, for ``rollout``."""
    pixels = np.asarray(pixels, np.float32)
    positions = np.asarray(positions, np.int64)
    states = np.zeros((len(pixels), *base.shape), np.float32)
    index = np.arange(len(pixels))
    states[index, :, positions[:, 0], positions[:, 1]] = np.asarray(base[:, training[0], training[1]], np.float32)
    states[index, : pixels.shape[1], positions[:, 0], positions[:, 1]] = pixels
    shifts = np.stack([training[0] - positions[:, 0], training[1] - positions[:, 1]], axis=-1).astype(np.int32)
    return states, shifts


def refine_alphas(alphas, labels, points):
    """``points`` new alphas evenly spread in every interval between neighbouring
    alphas whose labels differ (``alphas`` sorted)."""
    new = [np.linspace(alphas[i], alphas[i + 1], points + 2)[1:-1]
           for i in range(len(alphas) - 1) if labels[i] != labels[i + 1]]
    return np.concatenate(new) if new else np.zeros(0)


def adaptive_sweeps(evaluate, n_sweeps, coarse=21, rounds=3, points=4, label="final_class"):
    """Alphas for ``n_sweeps`` colour sweeps, denser where the outcome changes.

    Starts with ``coarse`` evenly spaced alphas per sweep; then ``rounds`` times
    adds ``points`` alphas in every interval whose two ends have a different
    ``label``. ``evaluate(sweep_index [M], alpha [M])`` runs all requested seeds
    at once and returns a dict of [M] arrays. Returns, per sweep, that dict for all
    its alphas (sorted), with ``alpha`` and ``round`` (0 = coarse) added.
    """
    results = [None] * n_sweeps
    sweep = np.repeat(np.arange(n_sweeps), coarse)
    alpha = np.tile(np.linspace(0.0, 1.0, coarse), n_sweeps)
    for round_ in range(rounds + 1):
        if not len(alpha):
            break
        out = evaluate(sweep, alpha)
        for s in np.unique(sweep):
            mask = sweep == s
            new = {name: np.asarray(value)[mask] for name, value in out.items()}
            new["alpha"], new["round"] = alpha[mask], np.full(mask.sum(), round_)
            if results[s] is not None:
                new = {name: np.concatenate([results[s][name], value]) for name, value in new.items()}
            order = np.argsort(new["alpha"], kind="stable")
            results[s] = {name: value[order] for name, value in new.items()}
        if round_ == rounds:
            break
        refined = [refine_alphas(results[s]["alpha"], results[s][label], points) for s in range(n_sweeps)]
        sweep = np.concatenate([np.full(len(r), s) for s, r in enumerate(refined)]).astype(int)
        alpha = np.concatenate(refined)
    return results


def boundary_windows(alphas, labels, half_width, limit=None):
    """Alpha ranges around every change of ``labels`` along a sorted sweep: from
    ``half_width`` before to ``half_width`` after the change, overlapping ranges
    merged; at most ``limit`` (the first ones)."""
    windows = []
    for i in range(len(alphas) - 1):
        if labels[i] != labels[i + 1]:
            middle = (alphas[i] + alphas[i + 1]) / 2
            low, high = max(0.0, middle - half_width), min(1.0, middle + half_width)
            if windows and low <= windows[-1][1]:
                windows[-1][1] = high
            else:
                windows.append([low, high])
    return windows[:limit] if limit else windows


def mse(a, b):
    """Mean squared error over the last three axes."""
    return np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=(-3, -2, -1))


def similarity(distances, targets):
    """Similarity to each target from the RGBA MSE ``distances`` [..., target]:
    1 - distance / distance of an empty grid to that target. 1 = identical to the
    target, 0 = as far from it as an empty grid (a seed is almost there), below 0 =
    further away than an empty grid."""
    empty = np.mean(np.asarray(targets, dtype=np.float64) ** 2, axis=(1, 2, 3))
    return 1.0 - np.asarray(distances, dtype=np.float64) / empty


def target_similarity(targets):
    """[k, j]: similarity of target image k to target j (1 on the diagonal), i.e.
    where each target lies in the similarity space."""
    distances = np.array([[mse(a, b) for b in targets] for a in targets])
    return similarity(distances, targets)


def moving_average(series, window):
    """Centred moving average along the last axis, edges repeated (window 1: unchanged)."""
    series = np.asarray(series, dtype=np.float64)
    if window <= 1 or series.shape[-1] <= 1:
        return series
    window = int(min(window, series.shape[-1]))
    left = window // 2
    widths = [(0, 0)] * (series.ndim - 1) + [(left, window - 1 - left)]
    padded = np.pad(series, widths, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1).mean(axis=-1)


def target_gap(targets):
    """Smallest distance between two different targets (inf for one target)."""
    gaps = [mse(targets[i], targets[j]) for i in range(len(targets)) for j in range(i + 1, len(targets))]
    return float(min(gaps)) if gaps else float("inf")


def classify(distances, final_rgba, threshold):
    """Label of each final state: the index of its target when the distance is
    below ``threshold``, else ``OTHER``; ``DIED`` when no cell is alive and
    ``DIVERGED`` for non-finite states.

    ``distances`` [..., target] are the final distances, ``final_rgba``
    [..., 4, H, W] the final states.
    """
    distances = np.asarray(distances)
    best = distances.min(axis=-1)
    labels = np.where(best < threshold, distances.argmin(axis=-1), OTHER)
    alive = np.nan_to_num(np.asarray(final_rgba)[..., 3, :, :], nan=1.0).max(axis=(-2, -1)) > ALIVE_ALPHA
    labels = np.where(alive, labels, DIED)
    return np.where(np.isfinite(best), labels, DIVERGED)


def outcome_name(label, names):
    return names[label] if label >= 0 else OUTCOME_NAMES[label]


def scan_class_name(index, names):
    """Name of a ``scan_class`` index, as in spatial_position_scan_combined.py."""
    index = int(index)
    if index < len(names):
        return names[index]
    return "INTERMEDIATE" if index == len(names) else "NO_VISIBLE_PATTERN"


def _smooth(series, window):
    """``smooth_series`` of the scan for each row of [N, steps]: centred moving median, edges repeated."""
    series = np.asarray(series, dtype=np.float64)
    if window <= 1 or series.shape[-1] <= 1:
        return series
    window = int(min(window, series.shape[-1]))
    left = window // 2
    padded = np.pad(series, ((0, 0), (left, window - 1 - left)), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window, axis=-1), axis=-1)


def _last_true(mask):
    """For each row: 1 + the index of the last True (the scan's 1-based step), 0 when there is none."""
    mask = np.asarray(mask, dtype=bool)
    last = mask.shape[-1] - np.argmax(mask[:, ::-1], axis=-1)
    return np.where(mask.any(axis=-1), last, 0)


def attractor_dynamics(classes, changes, n_targets):
    """The stable-attractor measures of spatial_position_scan_combined.py
    (``convergence_metrics``) for each run.

    ``classes`` [N, steps] are the ``scan_class`` after every step and
    ``changes`` [N, steps] the RGBA change per step (both from ``rollout(...,
    dynamics=True)``). Returns [N] arrays:

    - ``final_class``: the class after the last step
    - ``reaches_attractor``: the final class is a target, not INTERMEDIATE or
      NO_VISIBLE_PATTERN (``reaches_pure_attractor``)
    - ``steps_to_attractor``: the last step on which the class still differed
      from the final one (``steps_to_pure_attractor`` when it reaches a target)
    - ``settled``: the change came to rest before the end (``converged_before_end``)
    - ``steps_to_stable``: the last step on which the change was still above its
      threshold (``steps`` when it did not settle)
    """
    classes = np.asarray(classes)
    changes = np.asarray(changes, dtype=np.float64)
    total = classes.shape[-1]
    final = classes[:, -1]
    steps_to_attractor = _last_true(classes != final[:, None])

    series = _smooth(changes, CONVERGENCE_SMOOTH_WINDOW)
    residual = np.median(series[:, -min(CONVERGENCE_TAIL_STEPS, total):], axis=-1)
    peak = np.max(series, axis=-1)
    settled = (residual <= CONVERGENCE_SETTLED_FRACTION * peak) | (residual <= CONVERGENCE_ABS_TOL)
    threshold = np.maximum(CONVERGENCE_ABS_TOL, CONVERGENCE_REL_TOL * residual)
    steps_to_stable = np.where(settled, _last_true(series > threshold[:, None]), total)

    finite = np.all(np.isfinite(changes), axis=-1)
    return {
        "final_class": final,
        "reaches_attractor": (final < n_targets) & finite,
        "steps_to_attractor": steps_to_attractor,
        "settled": settled & (steps_to_stable < total) & finite,
        "steps_to_stable": steps_to_stable,
    }


def stable_attractor(dynamics, settled=False):
    """Index of the target each run reaches as a stable attractor, else -1.

    By default a run counts when it ends classified as a target
    (``reaches_pure_attractor`` of the scan); with ``settled`` its state must
    also have come to rest before the end (``converged_before_end``).
    """
    stable = dynamics["reaches_attractor"]
    if settled:
        stable = stable & dynamics["settled"]
    return np.where(stable, dynamics["final_class"], -1)


def evaluate_seeds(model, seeds, targets, pair_target, steps, horizon, repeats, key, tolerance, memory_gb=1.0):
    """Roll out every training seed ``repeats`` times and score the outcomes.

    Returns ``(rows, final, labels)``. ``rows`` has one row per pair with the
    fraction of rollouts that grew the right target (``correct``), grew a
    different target (``wrong_target``), grew something else (``other``) or
    died/diverged (``failed``), and the mean distance to the right target at
    ``horizon`` (the training rollout length) and at the end. ``final``
    [pair, repeat, 4, H, W] is the final RGBA of every rollout and ``labels``
    [pair, repeat] its outcome (see ``classify``).
    """
    states = np.repeat(seeds, repeats, axis=0)
    distances, rgba = rollout(model, states, targets, steps, key, memory_gb=memory_gb)
    labels = classify(distances[:, -1], rgba[:, -1], tolerance * target_gap(targets))
    rows = []
    for pair, target in enumerate(pair_target):
        own = slice(pair * repeats, (pair + 1) * repeats)
        pair_labels = labels[own]
        rows.append({
            "pair": pair,
            "target": int(target),
            "correct": float(np.mean(pair_labels == target)),
            "wrong_target": float(np.mean((pair_labels >= 0) & (pair_labels != target))),
            "other": float(np.mean(pair_labels == OTHER)),
            "failed": float(np.mean(pair_labels <= DIED)),
            "mse_at_t": float(np.mean(distances[own, min(horizon, steps), target])),
            "mse_final": float(np.mean(distances[own, -1, target])),
        })
    shape = (len(pair_target), repeats)
    return rows, rgba[:, -1].reshape(*shape, *rgba.shape[-3:]), labels.reshape(shape)


def to_image(rgba):
    """RGBA [..., 4, H, W] drawn over white, as [..., H, W, 3] in [0, 1]."""
    rgba = np.clip(np.nan_to_num(np.asarray(rgba)), 0.0, 1.0)
    rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:4, :, :]
    return np.moveaxis(rgb * alpha + (1.0 - alpha), -3, -1)
