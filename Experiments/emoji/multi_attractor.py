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

# Outcome labels and the nearest-target rule are shared with the impulse validation
from NCA.trainer.impulse.outcomes import ALIVE_ALPHA, DIED, DIVERGED, OTHER, OUTCOME_NAMES, classify, mse, target_gap  # noqa: F401

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


def _rollout(model, state, key, targets, steps, stride):
    """One trajectory of ``steps`` updates: RGBA every ``stride`` steps and the
    distance to each target at every step, both including the seed; and after
    every step the scan class and the RGBA change since the step before."""
    data_channels = targets.shape[1]

    def distance(state):
        return jnp.mean((state[None, :data_channels] - targets) ** 2, axis=(1, 2, 3))

    def step(state, index):
        new = model(state, lambda x: x, key=jax.random.fold_in(key, index))
        visible = new[:data_channels]
        change = jnp.mean((visible - state[:data_channels]) ** 2)
        return new, (distance(new), scan_class(visible, targets), change)

    def frame(state, start):
        state, outputs = jax.lax.scan(step, state, start + jnp.arange(stride))
        return state, (state[:data_channels], outputs)

    _, (frames, (distances, classes, changes)) = jax.lax.scan(frame, state, stride * jnp.arange(steps // stride))
    frames = jnp.concatenate([state[None, :data_channels], frames])
    distances = jnp.concatenate([distance(state)[None], distances.reshape(-1, len(targets))])
    return frames, distances, classes.reshape(-1), changes.reshape(-1)


@eqx.filter_jit
def _rollout_batch(model, states, keys, targets, steps, stride):
    return jax.vmap(lambda state, key: _rollout(model, state, key, targets, steps, stride))(states, keys)


def chunk_size(model, height, width, memory_gb):
    """How many seeds can be rolled out at once in about ``memory_gb`` of device memory.

    The largest arrays are the hidden layer's feature maps (``N_FEATURES`` x H x W
    floats per seed). A compiled rollout holds about three of them per seed;
    one more is allowed for the GPU convolution workspace.
    """
    features = getattr(model, "N_FEATURES", 4 * model.N_CHANNELS)
    per_seed = 4 * 4 * features * height * width
    return max(1, int(memory_gb * 2**30 // per_seed))


def rollout(model, states, targets, steps, key, frames=1, memory_gb=1.0, dynamics=False):
    """Roll out each seed in ``states`` [N, C, H, W] for ``steps`` updates.

    Returns the distance to each target at every step, [N, steps + 1, target],
    and the RGBA every ``steps / frames`` steps, [N, frames + 1, 4, H, W]
    (``frames=1``: the seed and the final state). With ``dynamics`` also the
    scan class (``scan_class``) and the RGBA change after every step, both
    [N, steps], for ``attractor_dynamics``. Only these are kept, and the
    seeds are run in chunks that fit in about ``memory_gb`` (see ``chunk_size``).
    """
    if steps % frames:
        raise ValueError(f"steps ({steps}) must be a multiple of frames ({frames})")
    states = np.asarray(states, dtype=np.float32)
    count = len(states)
    chunk = min(count, chunk_size(model, *states.shape[-2:], memory_gb))
    keys = jax.random.split(key, count)
    # Pad to whole chunks, so every chunk has the same shape and compiles once
    padding = -count % chunk
    states = np.concatenate([states, np.zeros((padding, *states.shape[1:]), np.float32)])
    keys = jnp.concatenate([keys, jnp.repeat(keys[:1], padding, axis=0)])
    targets = jnp.asarray(targets)
    # rgba, distances, classes, changes
    outputs = ([], [], [], [])
    for start in range(0, len(states), chunk):
        results = _rollout_batch(
            model, jnp.asarray(states[start : start + chunk]), keys[start : start + chunk],
            targets, int(steps), int(steps // frames),
        )
        for output, result in zip(outputs, results):
            output.append(np.asarray(result))
    rgba, distances, classes, changes = (np.concatenate(output)[:count] for output in outputs)
    if dynamics:
        return distances, rgba, classes, changes
    return distances, rgba


<<<<<<< HEAD
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


=======
>>>>>>> upstream/main
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
