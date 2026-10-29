# /// script
# dependencies = [
#   "marimo",
#   "anywidget",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

"""Explore emoji NCAs trained to grow several targets, one per seed, and how
many steps they need to reach a stable attractor.

Run from the repository root with:

    marimo edit Experiments/emoji/multi_attractor_steps_explorer.py

This is ``multi_attractor_explorer.py`` plus, from the scan and plot scripts
(``spatial_position_scan_combined.py``, ``spatial_position_scan_dynamic.py``,
``plot_steps_vs_spectrum_en.py``, ``plot_steps_vs_spectrum_dynamic.py``):

- the dynamic system (every run until its image is stable)
- the colour wheel and cube coloured by the steps to a stable attractor, for the
  normal and the dynamic system, with a colour scale
- steps vs. colour sweep plots for both systems, with more alphas near the
  boundaries between outcomes and zooms into them; optionally at all 9 positions

Model bundles are read from the model store (``MODEL_STORE_ROOT``, default
``models/``) and the emoji images from ``DATA_PATH_BASE/Emojis``. Only bundles
with ``data.emoji.task: multi_attractor`` are listed. The evaluation and
rollout helpers are in ``Experiments/emoji/multi_attractor_steps.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import itertools
    import math
    import os
    import zlib
    from pathlib import Path

    import anywidget
    import jax
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import traitlets
    from dotenv import load_dotenv
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Patch

    from Experiments.emoji import multi_attractor_steps as ma
    from Experiments.model_registry import ModelRegistry, open_model_bundle, verify_evaluation_input

    _repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(_repository_root / ".env", override=False)
    DEFAULT_STORE = os.environ.get("MODEL_STORE_ROOT", str(_repository_root / "models"))
    DEFAULT_DATA = os.environ.get("DATA_PATH_BASE", "")

    # Light figures regardless of the marimo theme
    plt.style.use("default")

    # Categorical hues for targets, in the order of data.emoji.pairs
    COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#8c564b")
    # Outcomes that are not a target
    OUTCOME_COLOURS = {ma.OTHER: "#c8c8c8", ma.DIED: "#3a3a3a", ma.DIVERGED: "#b8001f"}

    def outcome_colour(label):
        return COLOURS[label % len(COLOURS)] if label >= 0 else OUTCOME_COLOURS[label]

    def outcome_cmap(n_targets):
        """Colour map for ``label + 3``: diverged, died, other, then each target."""
        return ListedColormap([OUTCOME_COLOURS[ma.DIVERGED], OUTCOME_COLOURS[ma.DIED],
                               OUTCOME_COLOURS[ma.OTHER]] + [outcome_colour(k) for k in range(n_targets)])

    def outcome_legend(names):
        labels = list(range(len(names))) + [ma.OTHER, ma.DIED, ma.DIVERGED]
        return [Patch(color=outcome_colour(label), label=ma.outcome_name(label, names)) for label in labels]

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def show_image(ax, rgba, title=None, colour="black"):
        ax.imshow(ma.to_image(rgba), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=8, color=colour)

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def strip_plot(ax, df, x, y, hue):
        """One dot per model and the mean of each (x, hue) level, side by side for each hue."""
        df = df.copy()
        for column in filter(None, (x, hue)):
            df[column] = df[column].where(df[column].notna(), "none")
        x_order = levels(df[x])
        hue_order = levels(df[hue]) if hue else [None]
        width = 0.8 / len(hue_order)
        for k, level in enumerate(hue_order):
            rows = df if level is None else df[df[hue] == level]
            offset = (k - (len(hue_order) - 1) / 2) * width
            colour = COLOURS[k % len(COLOURS)]
            means = []
            for i, value in enumerate(x_order):
                values = rows.loc[rows[x] == value, y].dropna().to_numpy(float)
                jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(values)) * width
                ax.scatter(i + offset + jitter, values, s=16, color=colour, alpha=0.5, lw=0)
                means.append(values.mean() if len(values) else np.nan)
            ax.scatter(np.arange(len(x_order)) + offset, means, marker="_", s=400, lw=2.5, color=colour,
                       label=None if level is None else str(level))
        ax.set_xticks(range(len(x_order)), [str(v) for v in x_order])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        style_axes(ax)
        if hue:
            ax.legend(title=hue, frameon=False, fontsize=8)

    def outcome_gallery(labels, finals, captions, names, per_row=8):
        """Up to ``per_row`` final states of each outcome, spread over its members."""
        present = [int(label) for label in np.unique(labels)]
        fig, axes = plt.subplots(len(present), per_row, figsize=(1.3 * per_row, 1.5 * len(present)), squeeze=False)
        for row, label in enumerate(present):
            members = np.flatnonzero(labels == label)
            shown = members[np.linspace(0, len(members) - 1, min(per_row, len(members))).round().astype(int)]
            for col in range(per_row):
                if col < len(shown):
                    show_image(axes[row, col], finals[shown[col]], captions[shown[col]], outcome_colour(label))
                else:
                    axes[row, col].axis("off")
            axes[row, 0].set_ylabel(f"{ma.outcome_name(label, names)}\n({len(members)})",
                                    color=outcome_colour(label), fontsize=9)
        fig.tight_layout()
        return fig

    # Similarity space (velocity map): one axis per target. ``paths`` are dicts with
    # label, colour, style and sim [rollout, step, target] (``ma.similarity``); with
    # "small": True a path is drawn thin, with small markers and a dark outline (the many
    # start colours between the trained seeds).
    LINESTYLES = ("-", "--", ":", "-.")

    def symlog(values, c):
        """Symmetric logarithm sign(x) log10(1 + |x| / c): linear for |x| << c, logarithmic beyond."""
        values = np.asarray(values, dtype=float)
        return np.sign(values) * np.log10(1.0 + np.abs(values) / c)

    # Start axes: anchored at the lowest start point, so the start points sit at the bottom left.
    # Half of the axis is logarithmic in the offset from that point (the start points, which differ
    # only by about 1e-4, spread out), the other half linear (the course of the paths keeps its
    # shape further out). An axis is (anchor, c, top): offsets below c are linear, ``top`` is the
    # largest offset shown.
    START_LOG_SHARE = 0.5

    def start_axes(paths, magnify, low, high):
        """(anchor, c, top) for every target: anchor = the lowest start point, c = the spread of the
        start points / ``magnify``, top = the largest |offset| between the similarities low and high."""
        starts = np.concatenate([path["sim"][:, 0, :] for path in paths], axis=0)
        axes = {}
        for k in range(starts.shape[1]):
            values = starts[:, k][np.isfinite(starts[:, k])]
            anchor = float(values.min()) if values.size else 0.0
            spread = float(values.max() - values.min()) if values.size else 0.0
            c = spread / magnify if spread > 0 else max(abs(anchor), 1e-6) / magnify
            axes[k] = (anchor, max(c, 1e-12), max(abs(high[k] - anchor), abs(low[k] - anchor), 1e-9))
        return axes

    def start_scale(values, axis):
        """Position of similarities on a start axis: 0 = the lowest start point, 1 = ``top`` above it.
        Above the lowest start point half logarithmic, half linear; below it (paths that end up less
        similar than any start) linear only, with the slope of the linear half, so it takes little room."""
        anchor, c, top = axis
        offset = np.asarray(values, dtype=float) - anchor
        above = np.clip(offset, 0.0, None)
        logarithmic = np.log10(1.0 + above / c) / np.log10(1.0 + top / c)
        return np.where(offset >= 0, START_LOG_SHARE * logarithmic + (1 - START_LOG_SHARE) * above / top,
                        (1 - START_LOG_SHARE) * offset / top)

    def start_ticks(axis, low, high):
        """Ticks of a start axis between the positions low and high: near the start points the offsets
        from the lowest one (+1e-6, +1e-5, …), further out the similarities themselves (0.1, 0.5, 1)."""
        anchor, c, top = axis
        near = [10.0 ** m for m in range(math.ceil(math.log10(c)), -1)]
        candidates = [(0.0, "+0")] + [(d, f"+1e{round(math.log10(d))}") for d in near] \
            + [(v - anchor, f"{v:g}") for v in (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0)]
        ticks = sorted((float(start_scale(anchor + offset, axis)), label) for offset, label in candidates)
        kept = []
        for position, label in ticks:
            # Inside the range, and not closer than 4 % of it to the tick before
            if low <= position <= high and (not kept or position - kept[-1][0] > 0.04 * (high - low)):
                kept.append((position, label))
        return [position for position, _ in kept], [label for _, label in kept]

    def similarity_limits(paths, target_sim):
        """Common axis range: from 0 (or lower, if a path goes there) to just past 1."""
        values = np.concatenate([target_sim.ravel()] + [path["sim"].ravel() for path in paths])
        values = values[np.isfinite(values)]
        return (min(0.0, float(values.min())) if len(values) else 0.0) - 0.05, 1.1

    def path_arrows(sim, stride):
        """Start points and steps of arrows every ``stride`` steps along one path [step, target]."""
        start = np.arange(0, len(sim) - 1, stride)
        end = np.minimum(start + stride, len(sim) - 1)
        return sim[start], sim[end] - sim[start]

    def similarity_plane(ax, i, j, paths, target_sim, targets, names, stride, limits, log=None):
        """The plane of targets i (x) and j (y); each target's image sits at the end of its axis.

        Arrows go from the state at step t to the state at step t + ``stride``, so they chain
        along the path and their length is the distance covered in ``stride`` steps. With
        ``log`` ({target: (anchor, c, top)}, see ``start_axes``) both axes are start axes: the
        start points at the bottom left, logarithmic near them, linear further out; ``limits``
        is then {target: (low, high)} in similarities. The arrows still join the same steps, but
        their length is no longer proportional to the speed.
        """
        if log is None:
            fx = fy = (lambda v: np.asarray(v, dtype=float))
        else:
            fx = (lambda v: start_scale(v, log[i]))
            fy = (lambda v: start_scale(v, log[j]))
        for path in paths:
            small = path.get("small", False)
            for r, sim in enumerate(path["sim"]):
                ax.plot(fx(sim[:, i]), fy(sim[:, j]), color=path["colour"], ls=path["style"],
                        lw=0.9 if small else 1, alpha=0.8 if small else 0.45, zorder=2)
                at, step = path_arrows(sim, stride)
                x0, y0 = fx(at[:, i]), fy(at[:, j])
                ax.quiver(x0, y0, fx(at[:, i] + step[:, i]) - x0, fy(at[:, j] + step[:, j]) - y0, angles="xy",
                          scale_units="xy", scale=1, color=path["colour"], width=0.003 if small else 0.005,
                          headwidth=4, headlength=5, headaxislength=4.5, minlength=0, zorder=3,
                          edgecolor="#333333" if small else None, linewidth=0.3 if small else 0)
                # Start points larger on the log axes, which are there to show them
                size = (28 if small else 55) * (1.6 if log is not None else 1.0)
                ax.scatter(fx(sim[0, i]), fy(sim[0, j]), s=size, color=path["colour"], edgecolor="black",
                           lw=0.6 if small else 1.0, zorder=4)
                ax.scatter(fx(sim[-1, i]), fy(sim[-1, j]), s=22 if small else 45, marker="s", color=path["colour"],
                           edgecolor="black", lw=0.6 if small else 1.0, zorder=4)
                if log is not None and not small and r == 0 and path.get("label"):
                    ax.annotate(path["label"], (fx(sim[0, i]), fy(sim[0, j])), xytext=(6, -10),
                                textcoords="offset points", fontsize=7, color=path["colour"])
        for k, name in enumerate(names):
            ax.scatter(fx(target_sim[k, i]), fy(target_sim[k, j]), marker="*", s=260, color=outcome_colour(k),
                       edgecolor="black", lw=0.8, zorder=5)
            ax.annotate(name, (fx(target_sim[k, i]), fy(target_sim[k, j])), xytext=(7, 5),
                        textcoords="offset points", fontsize=8)
        for value in (0.0, 1.0):
            ax.axvline(fx(value), color="#bbbbbb", lw=0.8, ls="-" if value == 0 else "--", zorder=0)
            ax.axhline(fy(value), color="#bbbbbb", lw=0.8, ls="-" if value == 0 else "--", zorder=0)
        if log is None:
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal")
            ax.set_xlabel(f"similarity to {names[i]}")
            ax.set_ylabel(f"similarity to {names[j]}")
        else:
            ax.set_box_aspect(1)
            for k, set_limits, set_ticks, set_label, rotation in (
                    (i, ax.set_xlim, ax.set_xticks, ax.set_xlabel, 45), (j, ax.set_ylim, ax.set_yticks, ax.set_ylabel, 0)):
                low, high = start_scale(np.asarray(limits[k]), log[k])
                margin = 0.03 * (high - low)
                set_limits(low - margin, high + margin)
                positions, labels = start_ticks(log[k], low - margin, high + margin)
                set_ticks(positions, labels, fontsize=7, rotation=rotation)
                set_label(f"similarity to {names[k]}\n(near the start: + offset from {log[k][0]:.3g}, log; "
                          "further out linear)", fontsize=8)
            # The lowest start point on each axis
            ax.axvline(0.0, color="#d0d0d0", lw=0.6, ls=":", zorder=0)
            ax.axhline(0.0, color="#d0d0d0", lw=0.6, ls=":", zorder=0)
        style_axes(ax)
        # The target images at the ends of their axes: i right of the x axis, j on top of the y axis
        zoom = 30 / targets.shape[-1]
        for k, xy, align in ((i, (1.0, 0.0), (-0.15, 0.5)), (j, (0.0, 1.0), (0.5, -0.15))):
            box = AnnotationBbox(OffsetImage(ma.to_image(targets[k]), zoom=zoom), xy, xycoords="axes fraction",
                                 box_alignment=align, frameon=True, pad=0.1, annotation_clip=False,
                                 bboxprops={"edgecolor": outcome_colour(k), "lw": 1.5})
            ax.add_artist(box)

    def similarity_space_3d(ax, paths, target_sim, names, stride, limits):
        """The space of the first three targets."""
        for path in paths:
            small = path.get("small", False)
            for sim in path["sim"]:
                ax.plot(sim[:, 0], sim[:, 1], sim[:, 2], color=path["colour"], ls=path["style"],
                        lw=0.9 if small else 1, alpha=0.8 if small else 0.45)
                at, step = path_arrows(sim, stride)
                ax.quiver(at[:, 0], at[:, 1], at[:, 2], step[:, 0], step[:, 1], step[:, 2], color=path["colour"],
                          arrow_length_ratio=0.3, linewidth=0.8 if small else 1.2)
                ax.scatter(*sim[0, :3], s=28 if small else 55, color=path["colour"], edgecolor="black",
                           depthshade=False)
                ax.scatter(*sim[-1, :3], s=22 if small else 45, marker="s", color=path["colour"], edgecolor="black",
                           depthshade=False)
        for k, name in enumerate(names[:3]):
            ax.scatter(*target_sim[k, :3], marker="*", s=260, color=outcome_colour(k), edgecolor="black",
                       depthshade=False)
            ax.text(*target_sim[k, :3], f"  {name}", fontsize=8)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.set_xlabel(f"similarity to {names[0]}", fontsize=8)
        ax.set_ylabel(f"similarity to {names[1]}", fontsize=8)
        ax.set_zlabel(f"similarity to {names[2]}", fontsize=8)

    # Colour of the start points that reach no stable attractor
    NO_ATTRACTOR_COLOUR = "#9a9a9a"
    # Path of the specific start point
    SPECIAL_COLOUR = "#7b2cbf"

    def start_basin_background(ax, plane, i, j, n_targets, log=None):
        """Regions of start points in the plane of targets i and j, coloured by the stable
        attractor they reach (grey: none), with their boundaries. ``plane`` has sim
        [a, b, target] (similarity of each start point) and stable [a, b]. ``log``: as in
        ``similarity_plane``."""
        x, y, stable = plane["sim"][..., i], plane["sim"][..., j], plane["stable"]
        if log is not None:
            x, y = start_scale(x, log[i]), start_scale(y, log[j])
        for k in range(-1, n_targets):
            region = (stable == k).astype(float)
            if not region.any():
                continue
            colour = outcome_colour(k) if k >= 0 else NO_ATTRACTOR_COLOUR
            # Filled where the region's indicator is above 1/2, outlined there
            ax.contourf(x, y, region, levels=[0.5, 1.5], colors=[colour], alpha=0.28, zorder=0)
            if region.min() == 0:
                ax.contour(x, y, region, levels=[0.5], colors=[colour], linewidths=1.8, zorder=1)

    def start_basin_legend(names):
        return [Patch(color=outcome_colour(k), alpha=0.45, label=f"start → {name}") for k, name in enumerate(names)] + [
            Patch(color=NO_ATTRACTOR_COLOUR, alpha=0.45, label="start → no stable attractor")]

    def similarity_legend(paths):
        handles = [Line2D([], [], color=path["colour"], ls=path["style"], label=path["label"]) for path in paths
                   if path.get("label")]
        handles += [
            Line2D([], [], marker="o", ls="", color="#888888", markeredgecolor="black", label="start signal"),
            Line2D([], [], marker="s", ls="", color="#888888", markeredgecolor="black", label="end"),
            Line2D([], [], marker="*", ls="", color="#888888", markeredgecolor="black", markersize=12,
                   label="target image"),
        ]
        return handles

    # ------------------------------------------------------------------
    # PCA of the states: the same paths as in the velocity map, on principal components.
    # ``paths`` as above, with "pcs" [rollout, kept state, component] instead of "sim".

    def pca_limits(paths, targets, mode):
        """Range of every component: of all points ("all"), or of the start points with a margin
        ("start", to see how the start points lie)."""
        if mode == "start":
            points = np.concatenate([path["pcs"][:, 0] for path in paths])
            margin = 0.25
        else:
            points = np.concatenate([path["pcs"].reshape(-1, path["pcs"].shape[-1]) for path in paths]
                                    + ([targets] if targets is not None else []))
            margin = 0.06
        points = points[np.isfinite(points).all(axis=1)]
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high - low, 1e-9 * max(np.abs(points).max(), 1e-12))
        return np.stack([low - margin * span, high + margin * span], axis=-1)

    def pca_plane(ax, a, b, paths, explained, stride, limits, targets=None, names=None, zoom=False):
        """Components a (x) and b (y). Arrows join kept states ``stride`` apart, so their length is
        the distance moved in the (standardised) state space. ``targets`` [n, k]: target images as
        stars. ``zoom``: the axes are fitted to the start points and scaled independently; otherwise
        both axes have the same scale, unless one range is more than 4 times the other (then the
        plane is stretched to a square and the axis labels say so)."""
        for path in paths:
            small = path.get("small", False)
            for r, pcs in enumerate(path["pcs"]):
                ax.plot(pcs[:, a], pcs[:, b], color=path["colour"], ls=path["style"], lw=0.9 if small else 1,
                        alpha=0.8 if small else 0.45, zorder=2)
                at, step = path_arrows(pcs, stride)
                ax.quiver(at[:, a], at[:, b], step[:, a], step[:, b], angles="xy", scale_units="xy", scale=1,
                          color=path["colour"], width=0.003 if small else 0.005, headwidth=4, headlength=5,
                          headaxislength=4.5, minlength=0, zorder=3, edgecolor="#333333" if small else None,
                          linewidth=0.3 if small else 0)
                size = (28 if small else 55) * (1.6 if zoom else 1.0)
                ax.scatter(pcs[0, a], pcs[0, b], s=size, color=path["colour"], edgecolor="black",
                           lw=0.6 if small else 1.0, zorder=4)
                ax.scatter(pcs[-1, a], pcs[-1, b], s=22 if small else 45, marker="s", color=path["colour"],
                           edgecolor="black", lw=0.6 if small else 1.0, zorder=4)
                if zoom and not small and r == 0 and path.get("label"):
                    ax.annotate(path["label"], (pcs[0, a], pcs[0, b]), xytext=(6, -10), textcoords="offset points",
                                fontsize=7, color=path["colour"])
        if targets is not None:
            for k, name in enumerate(names):
                ax.scatter(targets[k, a], targets[k, b], marker="*", s=260, color=outcome_colour(k),
                           edgecolor="black", lw=0.8, zorder=5)
                ax.annotate(name, (targets[k, a], targets[k, b]), xytext=(7, 5), textcoords="offset points",
                            fontsize=8)
        ax.set_xlim(limits[a])
        ax.set_ylim(limits[b])
        # Same scale on both axes (true distances), unless one range is much smaller than the other
        spans = np.diff(limits[a])[0], np.diff(limits[b])[0]
        stretched = zoom or max(spans) > 4 * min(spans)
        if stretched:
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal")
        note = " – stretched" if stretched and not zoom else ""
        ax.set_xlabel(f"PC{a + 1} ({explained[a]:.1%} of the variance){note}")
        ax.set_ylabel(f"PC{b + 1} ({explained[b]:.1%} of the variance){note}")
        ax.axhline(0.0, color="#d8d8d8", lw=0.6, zorder=0)
        ax.axvline(0.0, color="#d8d8d8", lw=0.6, zorder=0)
        style_axes(ax)

    def pca_components_figure(pca, count=3):
        """What the first components are: where in the grid (loadings summed over the channels;
        red / blue = the state grows / shrinks there along the component) and which channels
        (share of the squared loadings; RGBA blue, hidden grey)."""
        count = min(count, len(pca["components"]))
        fig, axes = plt.subplots(2, count, figsize=(3.3 * count, 5.6), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.0, 0.55]})
        names = [("R", "G", "B", "A")[c] if c < 4 else f"h{c - 4}" for c in pca["channels"]]
        for k in range(count):
            component = pca["components"][k]
            spatial = component.sum(axis=0)
            limit = max(float(np.abs(spatial).max()), 1e-12)
            axes[0, k].imshow(spatial, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            axes[0, k].set_xticks([])
            axes[0, k].set_yticks([])
            axes[0, k].set_title(f"PC{k + 1} ({pca['explained'][k]:.1%}): where", fontsize=9)
            energy = (component ** 2).sum(axis=(1, 2))
            energy = energy / max(energy.sum(), 1e-300)
            axes[1, k].bar(range(len(energy)), energy,
                           color=["#2a78d6" if c < 4 else "#9a9a9a" for c in pca["channels"]])
            axes[1, k].set_xticks(range(len(energy)), names, fontsize=6, rotation=90)
            axes[1, k].set_title("which channels", fontsize=8)
            style_axes(axes[1, k])
        fig.tight_layout()
        return fig

    def _sides(first, second):
        """Colours of the two sides of a boundary: a stable one first, None for an unstable side."""
        first, second = (first, second) if first >= 0 else (second, first)
        return outcome_colour(int(first)), (outcome_colour(int(second)) if second >= 0 else None)

    def polar_boundary_lines(stable):
        """Lines between cells of a polar basin grid [hue, saturation ring] whose stable
        attractor (``ma.stable_attractor``, -1 = none) differs, for ``RGBCube.wheel_boundaries``.

        Cell (i, j) spans hue i/n .. (i+1)/n and saturation j/m .. (j+1)/m. Hue wraps
        around; a stable region at the rim is closed by an arc at saturation 1.
        """
        n, m = stable.shape
        lines = []
        for i in range(n):
            for j in range(m):
                here, ahead = stable[i, j], stable[(i + 1) % n, j]
                if here != ahead and max(here, ahead) >= 0:
                    lines.append([(i + 1) / n, j / m, (i + 1) / n, (j + 1) / m, *_sides(here, ahead)])
                outside = stable[i, j + 1] if j + 1 < m else -1
                if here != outside and max(here, outside) >= 0:
                    lines.append([i / n, (j + 1) / m, (i + 1) / n, (j + 1) / m, *_sides(here, outside)])
        return lines

    def cube_boundary_faces(stable):
        """Faces between neighbouring cells of an [n, n, n] RGB basin grid whose stable
        attractor differs, for ``RGBCube.cube_boundaries``. The walls of the cube are left open."""
        n = stable.shape[0]
        faces = []
        for axis in range(3):
            lower = np.take(stable, range(n - 1), axis=axis)
            upper = np.take(stable, range(1, n), axis=axis)
            for index in np.argwhere((lower != upper) & (np.maximum(lower, upper) >= 0)):
                centre = (index + 0.5) / n
                centre[axis] = (index[axis] + 1) / n
                faces.append([axis, *np.round(centre, 4).tolist(), 0.5 / n,
                              *_sides(lower[tuple(index)], upper[tuple(index)])])
        return faces

    # ------------------------------------------------------------------
    # Steps to a stable attractor. Colour scale of plot_steps_vs_spectrum_*.py: one
    # hue, light = few steps, dark = many; grey = no stable attractor.
    SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                  "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
    STEPS_CMAP = LinearSegmentedColormap.from_list("steps", SEQUENTIAL)

    def steps_range(*values):
        """Colour range as in the plot scripts (RGB_COLOUR_RANGE = "data"): smallest to
        largest value, rounded out to multiples of 8, at least 16 steps wide."""
        finite = np.concatenate([np.asarray(value, float).ravel() for value in values])
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return 0, 16
        low = int(np.floor(finite.min() / 8) * 8)
        return low, max(int(np.ceil((finite.max() + 1) / 8) * 8), low + 16)

    def steps_colours(steps, low, high):
        """Hex colour of every step count; NaN (no stable attractor) is grey."""
        steps = np.asarray(steps, float).ravel()
        rgba = STEPS_CMAP(np.clip((np.nan_to_num(steps) - low) / max(high - low, 1), 0.0, 1.0))
        return [to_hex(colour) if np.isfinite(value) else NO_ATTRACTOR_COLOUR for colour, value in zip(rgba, steps)]

    def field_steps(entry, system, settled, n_targets):
        """Steps to a stable attractor of every colour of a basin map or cube (NaN: none).

        normal: ``steps_to_pure_attractor`` of the colours that reach a stable attractor
        (criterion of the boundaries); dynamic: ``steps_to_stable_image`` of the colours
        whose stable image is a target. None when the dynamic system was not run.
        """
        if system == "normal":
            stable = ma.stable_attractor(entry["dynamics"], settled=settled)
            return np.where(stable >= 0, entry["dynamics"]["steps_to_attractor"], np.nan)
        dynamic = entry.get("dynamic")
        if dynamic is None:
            return None
        reached = dynamic["stable_image_reached"] & (dynamic["final_class"] < n_targets)
        return np.where(reached, dynamic["steps_to_stable_image"], np.nan)

    # ------------------------------------------------------------------
    # Steps vs. colour sweep plots (draw_spectrum_panel etc. of plot_steps_vs_spectrum_*.py)
    MARKERS = ("o", "s", "^", "v", "P", "X", "*", "h")
    INK_SECONDARY = "#52514e"
    AXIS_COLOUR = "#c3c2b7"
    GRID_COLOUR = "#e1e0d9"
    ZOOM_COLOUR = "#f1f0ea"
    POSITION_GRID = (("top_left", "top_center", "top_right"), ("center_left", "center", "center_right"),
                     ("bottom_left", "bottom_center", "bottom_right"))

    def sweep_class_style(index, n_targets, system):
        """(colour, marker, hollow) of a final class (``ma.scan_class``): targets in their
        colour; INTERMEDIATE a grey bar (normal, it has no steps) or diamond (dynamic);
        an empty canvas a grey ring."""
        if index < n_targets:
            return outcome_colour(index), MARKERS[index % len(MARKERS)], False
        if index == n_targets:
            return INK_SECONDARY, "|" if system == "normal" else "D", False
        return INK_SECONDARY, "o", True

    def sweep_class_name(index, names):
        return {len(names): "INTERMEDIATE", len(names) + 1: "empty canvas"}.get(int(index), ma.scan_class_name(index, names))

    def sweep_steps(result, system, settled):
        """y of every run: normal: ``steps_to_pure_attractor`` of runs that reach a stable
        attractor; dynamic: ``steps_to_stable_image`` of runs with a stable image (any
        class, also INTERMEDIATE). NaN: drawn in the band above the scale."""
        if system == "normal":
            stable = ma.stable_attractor(result, settled=settled)
            return np.where(stable >= 0, result["steps_to_attractor"], np.nan)
        return np.where(result["stable_image_reached"], result["steps_to_stable_image"], np.nan)

    def nice_step(upper, max_ticks=6):
        for step in (8, 16, 32, 64, 128, 256, 512, 1024):
            if upper / step <= max_ticks:
                return step
        return 2048

    def axis_upper(values, default=128):
        """Top of the y axis: the largest value, rounded up to the tick step (``axis_upper``)."""
        finite = np.concatenate([np.asarray(value, float).ravel() for value in values])
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return default
        step = nice_step(max(finite.max(), 1))
        return int(max(np.ceil((finite.max() + 1) / step) * step, step))

    def steps_panel(ax, strip, result, steps, upper, system, names, xlim=(0.0, 1.0), compact=False, ylabel=True,
                    zoom=False):
        """One colour sweep: a point per run at (alpha, steps), coloured and shaped by its final
        class; runs without steps in a band above the scale; the seed colours as a strip.
        compact: small labels; zoom: compact, but full-size points."""
        alpha = result["alpha"]
        size = 14 if compact else 30
        compact = compact or zoom
        pad = upper * (0.19 if compact else 0.125)
        band = upper + pad
        ax.plot(alpha, steps, color=AXIS_COLOUR, lw=1.0, zorder=1)
        for index in np.unique(result["final_class"]):
            mask = result["final_class"] == index
            colour, marker, hollow = sweep_class_style(int(index), len(names), system)
            y = np.where(np.isfinite(steps[mask]), steps[mask], band)
            if marker == "|":
                ax.scatter(alpha[mask], y, s=size * 2.2, marker="|", color=colour, linewidths=1.4, zorder=3)
            else:
                ax.scatter(alpha[mask], y, s=size * (0.8 if marker == "D" else 1.0), marker=marker,
                           facecolors="none" if hollow else colour, edgecolors=colour if hollow else "white",
                           linewidths=1.2 if hollow else 0.8, zorder=3)
        margin = 0.02 * (xlim[1] - xlim[0])
        ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
        ax.set_ylim(-upper * 0.05, band + pad * 0.6)
        ticks = list(range(0, int(upper) + 1, nice_step(upper)))
        none = "none" if compact else ("no stable\nattractor" if system == "normal" else "no stable\nimage")
        ax.set_yticks(ticks + [band], [str(t) for t in ticks] + [none])
        for tick in ticks:
            ax.axhline(tick, color=GRID_COLOUR, lw=0.8, zorder=0)
        ax.axhline(upper + pad / 2, color=AXIS_COLOUR, lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=0, labelsize=7 if compact else 8)
        if ylabel:
            ax.set_ylabel("steps until stable\nattractor" if system == "normal" else "step from which the\nimage is stable",
                          fontsize=7 if compact else 9)
        # Seed colours: each run's colour from midway to its neighbours
        edges = np.concatenate([[alpha[0]], (alpha[1:] + alpha[:-1]) / 2, [alpha[-1]]]) if len(alpha) else []
        strip.bar(edges[:-1], 1.0, width=np.diff(edges), align="edge", color=np.clip(result["rgb"], 0, 1), linewidth=0)
        strip.set_xlim(ax.get_xlim())
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        for side in strip.spines.values():
            side.set_visible(False)
        strip.tick_params(axis="x", length=2, color=AXIS_COLOUR, labelsize=7 if compact else 8)

    def sweep_legend(classes, names, system):
        handles = []
        for index in sorted(classes):
            colour, marker, hollow = sweep_class_style(int(index), len(names), system)
            if marker == "|":
                handles.append(Line2D([], [], ls="none", marker="|", markersize=9, markeredgewidth=1.4, color=colour,
                                      label=sweep_class_name(index, names)))
            else:
                handles.append(Line2D([], [], ls="none", marker=marker, markersize=6.5,
                                      markerfacecolor="none" if hollow else colour,
                                      markeredgecolor=colour if hollow else "white",
                                      markeredgewidth=1.2 if hollow else 0.8, label=sweep_class_name(index, names)))
        return handles

    def sweep_share(result, steps, system):
        share = np.isfinite(steps).mean()
        text = f"{share:.0%} of {len(steps)} runs reach a stable {'attractor' if system == 'normal' else 'image'}"
        if np.isfinite(steps).any():
            text += f", median step {np.nanmedian(steps):.0f}"
        return text

    def sweep_figure(result, steps, sweep, system, names, upper, half_width, position, max_zooms=4):
        """A colour sweep with zooms into the alpha ranges where the final class changes
        (``ma.boundary_windows``), which also have the most runs."""
        windows = ma.boundary_windows(result["alpha"], result["final_class"], half_width, limit=max_zooms)
        fig = plt.figure(figsize=(10.0, 4.8 + (3.4 if windows else 0.0)))
        outer = fig.add_gridspec(2 if windows else 1, 1, height_ratios=[1.4, 1.0] if windows else [1.0],
                                 hspace=0.42, left=0.11, right=0.98, top=0.83 if windows else 0.76, bottom=0.08)
        top = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.07], hspace=0.08)
        ax = fig.add_subplot(top[0])
        strip = fig.add_subplot(top[1], sharex=ax)
        steps_panel(ax, strip, result, steps, upper, system, names)
        strip.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        start, end = (names[sweep["a"]], names[sweep["b"]])
        if sweep["kind"] == "complementary":
            start, end = f"complement of {start}", f"complement of {end}"
        strip.set_xlabel(f"alpha (seed colour mix): {start} → {end}", fontsize=8)
        for k, (low, high) in enumerate(windows):
            ax.axvspan(low, high, color=ZOOM_COLOUR, zorder=0)
            ax.text((low + high) / 2, ax.get_ylim()[1], f"zoom {k + 1}", ha="center", va="bottom", fontsize=7,
                    color=INK_SECONDARY)
        if windows:
            bottom = outer[1].subgridspec(2, len(windows), height_ratios=[1.0, 0.1], hspace=0.08, wspace=0.28)
            for k, (low, high) in enumerate(windows):
                zax = fig.add_subplot(bottom[0, k])
                zstrip = fig.add_subplot(bottom[1, k], sharex=zax)
                inside = (result["alpha"] >= low) & (result["alpha"] <= high)
                # Also the nearest run on each side, so the line and the colour strip reach the edges
                index = np.flatnonzero(inside)
                index = np.arange(max(index[0] - 1, 0), min(index[-1] + 2, len(inside)))
                part = {name: value[index] for name, value in result.items()}
                steps_panel(zax, zstrip, part, steps[index], upper, system, names, xlim=(low, high), zoom=True,
                            ylabel=k == 0)
                zstrip.set_xticks([low, (low + high) / 2, high], [f"{low:.3f}", f"{(low + high) / 2:.3f}", f"{high:.3f}"])
                zax.set_title(f"zoom {k + 1}: {inside.sum()} runs", fontsize=8, color=INK_SECONDARY)
        fig.text(0.11, 0.965, f"{sweep['name']} · {position} · {system} system", fontsize=11)
        fig.text(0.11, 0.93, sweep_share(result, steps, system), fontsize=9, color=INK_SECONDARY)
        fig.legend(handles=sweep_legend(np.unique(result["final_class"]), names, system), loc="upper left",
                   bbox_to_anchor=(0.105, 0.925), ncol=len(np.unique(result["final_class"])), fontsize=8,
                   frameon=False, handletextpad=0.3, columnspacing=1.4)
        return fig

    def positions_figure(results, steps, sweep, system, names, upper):
        """The same sweep at all 9 positions, arranged as on the canvas (``plot_positions_grid``)."""
        fig = plt.figure(figsize=(11.0, 9.0))
        outer = fig.add_gridspec(3, 3, left=0.075, right=0.985, top=0.88, bottom=0.06, hspace=0.36, wspace=0.12)
        classes = set()
        for row, line in enumerate(POSITION_GRID):
            for column, position in enumerate(line):
                cell = outer[row, column].subgridspec(2, 1, height_ratios=[1.0, 0.09], hspace=0.06)
                ax = fig.add_subplot(cell[0])
                strip = fig.add_subplot(cell[1], sharex=ax)
                if position not in results:
                    ax.axis("off")
                    strip.axis("off")
                    continue
                steps_panel(ax, strip, results[position], steps[position], upper, system, names, compact=True,
                            ylabel=column == 0)
                strip.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
                share = np.isfinite(steps[position]).mean()
                ax.set_title(f"{position} · {share:.0%} stable", fontsize=8.5)
                classes |= set(np.unique(results[position]["final_class"]).tolist())
        fig.text(0.075, 0.955, f"{sweep['name']} · {system} system · all start positions", fontsize=11)
        fig.legend(handles=sweep_legend(classes, names, system), loc="upper left", bbox_to_anchor=(0.07, 0.94),
                   ncol=len(classes), fontsize=8, frameon=False)
        return fig

    class RGBCube(anywidget.AnyWidget):
        """A rotatable RGB cube: red, green (up) and blue along the axes, black at the origin.

        The far walls show their colours, so dots inside stay visible. Drag to
        rotate. With ``pickable`` a click on a wall chooses its colour and a click
        on a dot that dot's colour; sliders set any colour, including inner ones.
        The selection is sent when the mouse is released. ``markers`` are the
        training seed colours, ``points`` ``[r, g, b, colour, name]`` are drawn as
        small squares (the basin cube), described when hovered.

        With ``wheel`` a hue-saturation wheel at the chosen value is drawn beside
        the cube (hue is the angle, 0 = red to the right, anticlockwise;
        saturation the distance from the centre). Both show and set the same colour.

        ``wheel_boundaries`` and ``cube_boundaries`` outline the regions that reach
        a stable attractor (see ``polar_boundary_lines`` and ``cube_boundary_faces``),
        ``wheel_note`` says where the wheel's lines come from. ``wheel_cells`` and
        ``points`` colour the wheel and cube by steps, ``scale`` is their colour bar.
        """

        _esm = """
        // Cube rotations by view_key, kept when a cell re-renders the widget
        const views = (globalThis.__rgbCubeViews ??= {});
        const CORNERS = [[0, 0, 0, "K"], [1, 0, 0, "R"], [0, 1, 0, "G"], [0, 0, 1, "B"],
                         [0, 1, 1, "C"], [1, 0, 1, "M"], [1, 1, 0, "Y"], [1, 1, 1, "W"]];
        function rgbToHsv(r, g, b) {
          const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
          let h = 0;
          if (d > 0) {
            if (max === r) h = ((g - b) / d + 6) % 6;
            else if (max === g) h = (b - r) / d + 2;
            else h = (r - g) / d + 4;
          }
          return [h / 6, max > 0 ? d / max : 0, max];
        }
        function hsvToRgb(h, s, v) {
          const i = Math.floor(h * 6) % 6, f = h * 6 - Math.floor(h * 6);
          const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
          return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
        }
        function outlined(context, string, x, y) {
          context.lineWidth = 3;
          context.strokeStyle = "white";
          context.strokeText(string, x, y);
          context.fillStyle = "black";
          context.fillText(string, x, y);
        }
        function marker(context, x, y, colour, name) {
          context.beginPath();
          context.arc(x, y, 6, 0, 2 * Math.PI);
          context.fillStyle = colour;
          context.fill();
          context.lineWidth = 2;
          context.strokeStyle = "black";
          context.stroke();
          // Name on the left when it would run off the canvas
          const width = context.measureText(name).width;
          outlined(context, name, x + 9 + width > context.canvas.width ? x - 9 - width : x + 9, y + 4);
        }
        // Boundary lines [path, colour, other colour or null]: a white halo, the region's colour, and
        // dashes of the other side's colour where two stable regions meet. Layer by layer, so a
        // halo never covers the neighbouring segment's line.
        function boundaries(context, paths) {
          context.lineCap = "round";
          for (const [width, layer, dash] of [[4.5, () => "white", []], [2.5, (a) => a, []], [2.5, (a, b) => b, [4, 4]]]) {
            context.setLineDash(dash);
            context.lineWidth = width;
            for (const [path, a, b] of paths) {
              const stroke = layer(a, b);
              if (!stroke) continue;
              context.strokeStyle = stroke;
              path();
              context.stroke();
            }
          }
          context.setLineDash([]);
          context.lineCap = "butt";
        }
        function ring(context, x, y) {
          for (const [width, colour] of [[4, "white"], [2, "black"]]) {
            context.lineWidth = width;
            context.strokeStyle = colour;
            context.beginPath();
            context.arc(x, y, 9, 0, 2 * Math.PI);
            context.stroke();
          }
        }
        function rangeRow(label, value, onInput, onChange) {
          const row = document.createElement("label");
          row.style.cssText = "display: flex; gap: 8px; align-items: center; font: 12px sans-serif;";
          const input = document.createElement("input");
          Object.assign(input, { type: "range", min: 0, max: 1, step: 0.01, value });
          input.style.width = "200px";
          const caption = document.createElement("span");
          caption.style.width = "40px";
          caption.textContent = label;
          row.append(caption, input);
          input.addEventListener("input", () => onInput(parseFloat(input.value)));
          input.addEventListener("change", onChange);
          return [row, input];
        }
        function newCanvas(size, cursor) {
          const canvas = document.createElement("canvas");
          canvas.width = size;
          canvas.height = size;
          canvas.style.cssText = `cursor: ${cursor}; touch-action: none; max-width: 100%;`;
          return canvas;
        }
        function canvasPosition(canvas, size, event) {
          const box = canvas.getBoundingClientRect();
          return [(event.clientX - box.left) * size / box.width, (event.clientY - box.top) * size / box.height];
        }
        function caption(string) {
          const hint = document.createElement("div");
          hint.style.cssText = "font: 11px sans-serif; opacity: 0.7;";
          hint.textContent = string;
          return hint;
        }

        // Hue-saturation wheel at the chosen value: hue is the angle (0 = red, to the right,
        // anticlockwise), saturation the distance from the centre
        function wheelView(model, ui) {
          const size = 280, radius = 128, centre = size / 2;
          const element = document.createElement("div");
          const canvas = newCanvas(size, "crosshair"), context = canvas.getContext("2d");
          // Hue and saturation are kept here, as greys and black do not define them
          const hsv = rgbToHsv(model.get("r"), model.get("g"), model.get("b"));
          const [valueRow, valueInput] = rangeRow("value", hsv[2], (v) => {
            hsv[2] = v;
            ui.select(...hsvToRgb(...hsv));
          }, () => model.save_changes());
          const note = caption("");
          element.append(canvas, valueRow, caption("click or drag on the wheel, or click a dot"), note);
          let wheel = null, wheelValue = null;

          function sync() {
            const chosen = [model.get("r"), model.get("g"), model.get("b")];
            const local = hsvToRgb(...hsv);
            if (Math.abs(local[0] - chosen[0]) + Math.abs(local[1] - chosen[1]) + Math.abs(local[2] - chosen[2]) < 1e-4) return;
            const [h, s, v] = rgbToHsv(...chosen);
            if (v > 1e-6 && s > 1e-6) hsv[0] = h;
            if (v > 1e-6) hsv[1] = s;
            hsv[2] = v;
            valueInput.value = v;
          }
          function drawWheel() {
            wheel = context.createImageData(size, size);
            for (let y = 0; y < size; y++) {
              for (let x = 0; x < size; x++) {
                const dx = x - centre, dy = centre - y, r = Math.hypot(dx, dy) / radius;
                if (r > 1) continue;
                const h = (Math.atan2(dy, dx) / (2 * Math.PI) + 1) % 1;
                const rgb = hsvToRgb(h, r, hsv[2]), index = 4 * (y * size + x);
                wheel.data.set([255 * rgb[0], 255 * rgb[1], 255 * rgb[2], 255], index);
              }
            }
            wheelValue = hsv[2];
          }
          const position = (h, s) => [centre + s * radius * Math.cos(2 * Math.PI * h),
                                      centre - s * radius * Math.sin(2 * Math.PI * h)];
          function draw() {
            sync();
            if (wheelValue !== hsv[2]) drawWheel();
            context.clearRect(0, 0, size, size);
            context.putImageData(wheel, 0, 0);
            context.font = "11px sans-serif";
            // Steps colouring: annular cells [h0, h1, s0, s1, colour] over the wheel's own colours
            for (const [h0, h1, s0, s1, colour] of model.get("wheel_cells")) {
              context.beginPath();
              context.arc(centre, centre, s1 * radius, -2 * Math.PI * h0, -2 * Math.PI * h1, true);
              context.arc(centre, centre, s0 * radius, -2 * Math.PI * h1, -2 * Math.PI * h0, false);
              context.closePath();
              context.fillStyle = colour;
              context.fill();
              // The same colour as a hairline, so no seams show between cells
              context.lineWidth = 0.7;
              context.strokeStyle = colour;
              context.stroke();
            }
            // Boundaries of the stable-attractor regions: [h0, s0, h1, s1, colour, other side's colour or null],
            // an arc when s0 == s1, else a radial line
            const paths = model.get("wheel_boundaries").map(([h0, s0, h1, s1, a, b]) => [() => {
              context.beginPath();
              if (s0 === s1) {
                context.arc(centre, centre, s0 * radius, -2 * Math.PI * h0, -2 * Math.PI * h1, true);
              } else {
                context.moveTo(...position(h0, s0));
                context.lineTo(...position(h1, s1));
              }
            }, a, b]);
            boundaries(context, paths);
            note.textContent = model.get("wheel_note");
            note.style.display = note.textContent ? "" : "none";
            for (const m of model.get("markers")) {
              // Each seed at its own value
              const [x, y] = position(...rgbToHsv(m.r, m.g, m.b));
              marker(context, x, y, m.colour, m.label);
            }
            ring(context, ...position(hsv[0], hsv[1]));
          }
          function pick(event) {
            const [x, y] = canvasPosition(canvas, size, event);
            for (const m of model.get("markers")) {
              const [mx, my] = position(...rgbToHsv(m.r, m.g, m.b));
              if (Math.hypot(x - mx, y - my) < 8) {
                ui.select(m.r, m.g, m.b, m.a);
                return;
              }
            }
            const dx = x - centre, dy = centre - y;
            hsv[0] = (Math.atan2(dy, dx) / (2 * Math.PI) + 1) % 1;
            hsv[1] = Math.min(1, Math.hypot(dx, dy) / radius);
            ui.select(...hsvToRgb(...hsv));
          }
          let dragging = false;
          canvas.addEventListener("pointerdown", (event) => {
            dragging = true;
            canvas.setPointerCapture(event.pointerId);
            pick(event);
          });
          canvas.addEventListener("pointermove", (event) => { if (dragging) pick(event); });
          canvas.addEventListener("pointerup", () => {
            if (!dragging) return;
            dragging = false;
            model.save_changes();
          });
          return { element, draw };
        }

        // RGB cube: x = red, y = green (up), z = blue. The far walls are drawn, so dots inside stay visible.
        function cubeView(model, ui) {
          const size = 340, centre = size / 2, scale = (size / 2 - 24) / Math.sqrt(0.75);
          const view = (views[model.get("view_key")] ??= { yaw: 0.55, pitch: 0.4 });
          const pickable = model.get("pickable");
          const element = document.createElement("div");
          const canvas = newCanvas(size, "grab"), context = canvas.getContext("2d");
          element.append(canvas, caption(pickable ? "drag to rotate · click a wall or dot to choose"
                                                  : "drag to rotate · hover a dot for its colour and outcome"));

          // view = Rx(pitch) Ry(yaw) (world - 0.5)
          let ex, ey, ez;
          function basis() {
            const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
            const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
            ex = [cy, 0, sy];
            ey = [sp * sy, cp, -sp * cy];
            ez = [-cp * sy, sp, cp * cy];
          }
          function project(r, g, b) {
            const w = [r - 0.5, g - 0.5, b - 0.5];
            const dot = (u) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
            return [centre + scale * dot(ex), centre - scale * dot(ey), dot(ez)];
          }

          // Each pixel shows the colour where its view ray leaves the cube
          let walls = null, wallsView = null;
          function wallPoint(px, py) {
            const vx = (px - centre) / scale, vy = (centre - py) / scale;
            let lo = -Infinity, hi = Infinity, axis = 0;
            const base = [0, 0, 0];
            for (let i = 0; i < 3; i++) {
              base[i] = vx * ex[i] + vy * ey[i];
              const d = ez[i];
              if (Math.abs(d) < 1e-9) {
                if (Math.abs(base[i]) > 0.5) return null;
                continue;
              }
              const t1 = (-0.5 - base[i]) / d, t2 = (0.5 - base[i]) / d;
              const near = Math.min(t1, t2), far = Math.max(t1, t2);
              if (near > lo) { lo = near; axis = i; }
              if (far < hi) hi = far;
            }
            if (lo > hi) return null;
            const p = [0, 1, 2].map((i) => Math.min(1, Math.max(0, base[i] + lo * ez[i] + 0.5)));
            return [p, axis];
          }
          function drawWalls() {
            walls = context.createImageData(size, size);
            for (let y = 0; y < size; y++) {
              for (let x = 0; x < size; x++) {
                const hit = wallPoint(x + 0.5, y + 0.5);
                if (hit === null) continue;
                const [p, axis] = hit, shade = 0.8 + 0.2 * Math.abs(ez[axis]), index = 4 * (y * size + x);
                walls.data.set([255 * shade * p[0], 255 * shade * p[1], 255 * shade * p[2], 255], index);
              }
            }
            wallsView = `${view.yaw},${view.pitch}`;
          }

          let projected = [];
          function draw() {
            basis();
            if (wallsView !== `${view.yaw},${view.pitch}`) drawWalls();
            context.clearRect(0, 0, size, size);
            context.putImageData(walls, 0, 0);
            context.font = "11px sans-serif";

            // Edges, then corner names pushed away from the centre
            context.lineWidth = 1;
            context.strokeStyle = "rgba(60, 60, 60, 0.6)";
            for (const a of CORNERS) {
              for (const b of CORNERS) {
                const diff = Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]);
                if (diff !== 1 || a[3] > b[3]) continue;
                const [x1, y1] = project(a[0], a[1], a[2]), [x2, y2] = project(b[0], b[1], b[2]);
                context.beginPath();
                context.moveTo(x1, y1);
                context.lineTo(x2, y2);
                context.stroke();
              }
            }
            for (const [r, g, b, name] of CORNERS) {
              const [x, y] = project(r, g, b);
              const dx = x - centre, dy = y - centre, norm = Math.hypot(dx, dy) || 1;
              outlined(context, name, x + 12 * dx / norm - 4, y + 12 * dy / norm + 4);
            }

            // Points and markers, far to near
            const side = model.get("point_size");
            projected = [];
            for (const [r, g, b, colour, name] of model.get("points")) {
              projected.push({ kind: "point", r, g, b, colour, name, xyz: project(r, g, b) });
            }
            for (const m of model.get("markers")) {
              projected.push({ kind: "marker", ...m, xyz: project(m.r, m.g, m.b) });
            }
            // Boundaries of the stable-attractor regions: square faces between grid cells,
            // [axis, r, g, b (face centre), half side, colour, other side's colour or null]
            for (const [axis, r, g, b, half, colour, other] of model.get("cube_boundaries")) {
              const u = (axis + 1) % 3, v = (axis + 2) % 3;
              const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([du, dv]) => {
                const p = [r, g, b];
                p[u] += du * half;
                p[v] += dv * half;
                return project(...p);
              });
              projected.push({ kind: "face", corners, colour, other, xyz: project(r, g, b) });
            }
            projected.sort((p, q) => p.xyz[2] - q.xyz[2]);
            for (const p of projected) {
              const [x, y] = p.xyz;
              if (p.kind === "face") {
                context.beginPath();
                context.moveTo(p.corners[0][0], p.corners[0][1]);
                for (const [cx, cy] of p.corners.slice(1)) context.lineTo(cx, cy);
                context.closePath();
                context.globalAlpha = 0.35;
                context.fillStyle = p.colour;
                context.fill();
                context.globalAlpha = 0.8;
                context.lineWidth = p.other ? 1.5 : 0.5;
                context.strokeStyle = p.other ?? p.colour;
                context.stroke();
                context.globalAlpha = 1;
              } else if (p.kind === "point") {
                context.fillStyle = p.colour;
                context.fillRect(x - side / 2, y - side / 2, side, side);
                context.lineWidth = 0.5;
                context.strokeStyle = "rgba(0, 0, 0, 0.6)";
                context.strokeRect(x - side / 2, y - side / 2, side, side);
              } else {
                marker(context, x, y, p.colour, p.label);
              }
            }
            const [x, y] = project(model.get("r"), model.get("g"), model.get("b"));
            ring(context, x, y);
          }
          function under(x, y) {
            // Nearest object drawn on top
            for (let i = projected.length - 1; i >= 0; i--) {
              if (projected[i].kind === "face") continue;
              const p = projected[i], reach = p.kind === "marker" ? 8 : Math.max(4, model.get("point_size") / 2 + 1);
              if (Math.hypot(x - p.xyz[0], y - p.xyz[1]) < reach) return p;
            }
            return null;
          }
          function pick(x, y) {
            const p = under(x, y);
            if (p !== null) {
              ui.select(p.r, p.g, p.b, p.kind === "marker" ? p.a : undefined);
            } else {
              const hit = wallPoint(x, y);
              if (hit === null) return;
              ui.select(...hit[0]);
            }
            model.save_changes();
          }

          let drag = null, hovered = null, frame = null;
          canvas.addEventListener("pointerdown", (event) => {
            const start = canvasPosition(canvas, size, event);
            drag = { start, last: start, moved: false };
            canvas.setPointerCapture(event.pointerId);
          });
          canvas.addEventListener("pointermove", (event) => {
            const [x, y] = canvasPosition(canvas, size, event);
            if (drag !== null) {
              if (Math.hypot(x - drag.start[0], y - drag.start[1]) > 3) drag.moved = true;
              if (!drag.moved) return;
              view.yaw += 0.01 * (x - drag.last[0]);
              view.pitch = Math.max(-1.5, Math.min(1.5, view.pitch + 0.01 * (y - drag.last[1])));
              drag.last = [x, y];
              canvas.style.cursor = "grabbing";
              if (frame === null) frame = requestAnimationFrame(() => { frame = null; draw(); });
              return;
            }
            const p = under(x, y);
            if (p === hovered) return;
            hovered = p;
            ui.hover(p === null ? null
              : [p.r, p.g, p.b, p.a, p.kind === "marker" ? `seed → ${p.label}` : `→ ${p.name}`]);
          });
          canvas.addEventListener("pointerup", (event) => {
            if (drag === null) return;
            const moved = drag.moved;
            drag = null;
            canvas.style.cursor = "grab";
            if (!moved && pickable) pick(...canvasPosition(canvas, size, event));
          });
          canvas.addEventListener("pointerleave", () => {
            if (hovered === null) return;
            hovered = null;
            ui.hover(null);
          });
          return { element, draw };
        }

        function render({ model, el }) {
          let hovered = null, frame = null;
          const ui = {
            // Set the chosen colour (alpha only when given); the caller saves it
            select(r, g, b, a) {
              model.set("r", r);
              model.set("g", g);
              model.set("b", b);
              if (a !== undefined) model.set("a", a);
              schedule();
            },
            // [r, g, b, a, text] of the dot under the mouse, or null
            hover(info) {
              hovered = info;
              describe();
            },
          };
          const panels = [cubeView(model, ui)];
          if (model.get("wheel")) panels.unshift(wheelView(model, ui));
          const row = document.createElement("div");
          row.style.cssText = "display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;";
          row.append(...panels.map((panel) => panel.element));
          el.append(row);

          const sliders = {};
          if (model.get("pickable")) {
            for (const [name, label] of [["r", "red"], ["g", "green"], ["b", "blue"], ["a", "alpha"]]) {
              const [sliderRow, input] = rangeRow(label, model.get(name), (value) => {
                model.set(name, value);
                schedule();
              }, () => model.save_changes());
              sliders[name] = input;
              el.append(sliderRow);
            }
          }
          const readout = document.createElement("div");
          readout.style.cssText = "font: 12px monospace; display: flex; gap: 8px; align-items: center; margin-top: 4px;";
          const swatch = document.createElement("span");
          swatch.style.cssText = "display: inline-block; width: 28px; height: 18px; border: 1px solid #888;";
          const text = document.createElement("span");
          readout.append(swatch, text);
          // Colour scale of the steps colouring: {title, low, high, colours, missing, missing_label}
          const legend = document.createElement("div");
          legend.style.cssText = "font: 12px sans-serif; display: flex; gap: 8px; align-items: center; margin-top: 6px; flex-wrap: wrap;";
          el.append(readout, legend);

          function drawScale() {
            const scale = model.get("scale") ?? {};
            legend.replaceChildren();
            if (!scale.colours) {
              legend.style.display = "none";
              return;
            }
            legend.style.display = "flex";
            const span = (content, css = "") => {
              const item = document.createElement("span");
              item.textContent = content;
              item.style.cssText = css;
              return item;
            };
            const bar = span("", `display: inline-block; width: 220px; height: 12px; border: 1px solid #888;
              background: linear-gradient(to right, ${scale.colours.join(", ")});`);
            const missing = span("", `display: inline-block; width: 14px; height: 12px; border: 1px solid #888;
              background: ${scale.missing}; margin-left: 10px;`);
            legend.append(span(scale.title, "font-weight: 600;"), span(String(scale.low)), bar, span(String(scale.high)),
                          missing, span(scale.missing_label));
          }

          function describe() {
            const [r, g, b, a, note] = hovered ?? [model.get("r"), model.get("g"), model.get("b"), model.get("a"), "chosen"];
            swatch.style.background = `rgba(${[r, g, b].map((c) => Math.round(255 * c)).join(",")},${a ?? 1})`;
            const [h, s, v] = rgbToHsv(r, g, b);
            text.textContent = `R ${r.toFixed(2)}  G ${g.toFixed(2)}  B ${b.toFixed(2)}`
              + (a === undefined ? "" : `  A ${a.toFixed(2)}`)
              + `  |  H ${h.toFixed(3)}  S ${s.toFixed(2)}  V ${v.toFixed(2)}  |  ${note}`;
          }
          function drawAll() {
            frame = null;
            for (const [name, input] of Object.entries(sliders)) input.value = model.get(name);
            for (const panel of panels) panel.draw();
            describe();
          }
          function schedule() {
            if (frame === null) frame = requestAnimationFrame(drawAll);
          }
          for (const name of ["r", "g", "b", "a", "markers", "points", "point_size", "wheel_boundaries", "wheel_note",
                              "cube_boundaries", "wheel_cells"]) {
            model.on(`change:${name}`, schedule);
          }
          model.on("change:scale", drawScale);
          drawScale();
          drawAll();
        }
        export default { render };
        """
        r = traitlets.Float(1.0).tag(sync=True)
        g = traitlets.Float(0.0).tag(sync=True)
        b = traitlets.Float(0.0).tag(sync=True)
        a = traitlets.Float(1.0).tag(sync=True)
        # [{r, g, b, a, label, colour}]
        markers = traitlets.List([]).tag(sync=True)
        # [[r, g, b, colour, name]]
        points = traitlets.List([]).tag(sync=True)
        point_size = traitlets.Float(6.0).tag(sync=True)
        pickable = traitlets.Bool(True).tag(sync=True)
        # Widgets with the same key share their rotation
        view_key = traitlets.Unicode("picker").tag(sync=True)
        wheel = traitlets.Bool(False).tag(sync=True)
        # [[h0, s0, h1, s1, colour, other colour or None]]
        wheel_boundaries = traitlets.List([]).tag(sync=True)
        wheel_note = traitlets.Unicode("").tag(sync=True)
        # [[axis, r, g, b, half side, colour, other colour or None]]
        cube_boundaries = traitlets.List([]).tag(sync=True)
        # Wheel cells drawn over the wheel's colours (steps colouring): [[h0, h1, s0, s1, colour]]
        wheel_cells = traitlets.List([]).tag(sync=True)
        # Colour scale under the widget: {title, low, high, colours, missing, missing_label}, {} = none
        scale = traitlets.Dict({}).tag(sync=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Multi-attractor emoji NCAs

    - Each model grows a different emoji from each training seed (`data.emoji.pairs`)
    - **Models**: filter the multi-attractor bundles in the model store
    - **Evaluation**: roll out every training seed; does it grow the right emoji, and does it stay?
    - **Input space**: choose any seed colour, and map which emoji each colour grows into
    - A final state counts as a target when its RGBA mean squared error is below *tolerance* × the smallest error between two targets
    """)
    return


@app.cell(hide_code=True)
def _():
    _devices = ", ".join(f"{_device.platform}: {_device.device_kind}" for _device in jax.devices())
    mo.callout(
        f"JAX backend: **{jax.default_backend()}** ({_devices}). The first rollout of each model size includes compilation.",
        kind="success" if jax.default_backend() == "gpu" else "warn",
    )
    return


@app.cell(hide_code=True)
def _():
    store_root = mo.ui.text(DEFAULT_STORE, label="Model store", full_width=True)
    data_root = mo.ui.text(DEFAULT_DATA, label="Data root (holds Emojis/)", full_width=True)
    reload_button = mo.ui.button(label="Reload models", value=0, on_click=lambda count: count + 1)
    mo.vstack([store_root, data_root, mo.hstack([reload_button, mo.md("after copying new bundles into the store")],
                                                justify="start")])
    return data_root, reload_button, store_root


@app.cell(hide_code=True)
def _(reload_button, store_root):
    # Bundles are read from their folders, so the registry index does not need rebuilding
    reload_button.value
    _registry = ModelRegistry(Path(store_root.value).expanduser())
    bundles = {}
    _skipped = 0
    for _path in _registry.bundle_paths():
        try:
            _bundle = open_model_bundle(_path)
        except (ValueError, TypeError, KeyError):
            _skipped += 1
            continue
        if ma.is_multi_attractor(_bundle.config):
            bundles[_bundle.id] = _bundle
    runs = pd.DataFrame([ma.summary_row(_bundle) for _bundle in bundles.values()])
    if len(runs):
        runs = runs.sort_values("created_at", ascending=False).reset_index(drop=True)
    _skipped_text = f" ({_skipped} bundles could not be read)" if _skipped else ""
    mo.md(f"**{len(runs)}** multi-attractor bundles in `{_registry.root}`{_skipped_text}.")
    return bundles, runs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Models
    """)
    return


@app.cell(hide_code=True)
def _(runs):
    mo.stop(len(runs) == 0, mo.callout("No multi-attractor bundles found in this model store.", kind="warn"))

    def _label(value):
        # pandas 3: astype(str) keeps missing values as NaN, which breaks sorting
        return "None" if value is None or (isinstance(value, float) and value != value) else str(value)

    def _choice(column, label):
        _options = sorted(set(runs[column].map(_label)))
        return mo.ui.multiselect(_options, value=_options, label=label)

    _emojis = sorted({_name for _targets in runs.targets for _name in _targets.split(" + ")})
    model_filters = mo.ui.dictionary({
        "collection": _choice("collection", "Collection"),
        "wandb_group": _choice("wandb_group", "W&B group"),
        "attractors": _choice("attractors", "Attractors"),
        "family": _choice("family", "Family"),
        "padding": _choice("padding", "Padding"),
        "stability_mode": _choice("stability_mode", "Stability mode"),
    })
    target_filter = mo.ui.multiselect(_emojis, label="Must include targets")
    text_filter = mo.ui.text(label="Search model id / targets", placeholder="e.g. crab + lizard")
    mo.vstack([
        mo.hstack([model_filters[_name] for _name in ("collection", "wandb_group", "attractors")], justify="start", wrap=True),
        mo.hstack([model_filters[_name] for _name in ("family", "padding", "stability_mode")], justify="start", wrap=True),
        mo.hstack([target_filter, text_filter], justify="start", wrap=True),
    ])
    return model_filters, target_filter, text_filter


@app.cell(hide_code=True)
def _(model_filters, runs, target_filter, text_filter):
    _keep = np.ones(len(runs), dtype=bool)
    def _label(value):
        # pandas 3: astype(str) keeps missing values as NaN, which breaks sorting
        return "None" if value is None or (isinstance(value, float) and value != value) else str(value)

    for _column, _values in model_filters.value.items():
        _keep &= runs[_column].map(_label).isin(_values).to_numpy()
    for _name in target_filter.value:
        _keep &= runs.targets.apply(lambda _targets: _name in _targets.split(" + ")).to_numpy()
    if text_filter.value.strip():
        _text = text_filter.value.strip().lower()
        _keep &= (runs.model_id.str.lower().str.contains(_text, regex=False)
                  | runs.targets.str.lower().str.contains(_text, regex=False)).to_numpy()
    candidates = runs[_keep].reset_index(drop=True)
    model_table = mo.ui.table(
        candidates.dropna(axis=1, how="all").round(5),
        initial_selection=list(range(len(candidates))),
        page_size=10,
        label="Untick rows to drop models from the evaluation and from the input-space model list",
    )
    model_table
    return candidates, model_table


@app.cell(hide_code=True)
def _(candidates, model_table):
    _ticked = set(model_table.value["model_id"]) if len(model_table.value) else set()
    selected = candidates[candidates.model_id.isin(_ticked)].reset_index(drop=True)
    mo.md(f"**{len(selected)}** of {len(candidates)} models selected.")
    return (selected,)


@app.cell(hide_code=True)
def _(bundles, data_root):
    # Models and training seeds, loaded once per session
    _models = {}
    _seeds = {}

    def get_model(model_id, parameter_noise):
        if (model_id, parameter_noise) not in _models:
            _models[model_id, parameter_noise] = ma.load_model(bundles[model_id], parameter_noise)
        return _models[model_id, parameter_noise]

    def get_seeds(model_id):
        """(seeds, targets, pair_target, fingerprint warning or None) of a model."""
        if model_id not in _seeds:
            _bundle = bundles[model_id]
            _channels = get_model(model_id, True).N_CHANNELS
            _impath = Path(data_root.value).expanduser() / "Emojis"
            _seed, _target, _pair_target, _data = ma.load_seeds_and_targets(_bundle.config, _channels, _impath)
            _warning = None
            _provenance = _bundle.manifest.get("evaluation_input")
            if _provenance is not None:
                try:
                    verify_evaluation_input(_data, _provenance)
                except ValueError as _error:
                    _warning = str(_error)
            _seeds[model_id] = (_seed, _target, _pair_target, _warning)
        return _seeds[model_id]

    return get_model, get_seeds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation

    - Every training seed is rolled out *rollouts* times for *length* × `run.t` steps
    - `correct`: fraction of rollouts that end on the seed's own target; `wrong_target`: on another target
    - `mse_at_t`: error to the own target after `run.t` steps (the training length); `mse_final`: at the end, so lengths > 1 test whether the target stays
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_length = mo.ui.slider(1, 8, value=2, step=1, label="Length (× run.t)", show_value=True)
    eval_repeats = mo.ui.number(1, 64, value=4, step=1, label="Rollouts per seed")
    tolerance = mo.ui.slider(0.05, 1.0, value=0.25, step=0.05, label="Tolerance", show_value=True)
    parameter_noise = mo.ui.checkbox(value=True, label="Parameter noise (as in training)")
    eval_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    # Seeds are rolled out in chunks that fit this budget; lower it if the GPU runs out of memory.
    # It changes speed only, not results.
    memory_budget = mo.ui.number(0.125, 64, value=1.0, step=0.125, label="Rollout memory (GB)")
    eval_run = mo.ui.run_button(label="Evaluate selected models")
    mo.hstack([eval_length, eval_repeats, tolerance, parameter_noise, eval_seed, memory_budget, eval_run],
              justify="start", gap=1.5, wrap=True)
    return (
        eval_length,
        eval_repeats,
        eval_run,
        eval_seed,
        memory_budget,
        parameter_noise,
        tolerance,
    )


@app.cell
def _():
    # Scores of every model evaluated in this session, keyed by (model_id, settings),
    # so pressing the button again only rolls out new models
    score_cache = {}
    return (score_cache,)


@app.cell(hide_code=True)
def _(
    bundles,
    eval_length,
    eval_repeats,
    eval_run,
    eval_seed,
    get_model,
    get_seeds,
    memory_budget,
    parameter_noise,
    score_cache,
    selected,
    tolerance,
):
    _settings = (eval_length.value, eval_repeats.value, tolerance.value, parameter_noise.value, eval_seed.value)
    _todo = [_id for _id in selected.model_id if (_id, _settings) not in score_cache]
    _failures = []
    if eval_run.value and _todo:
        for _id in mo.status.progress_bar(_todo, title="Rolling out models"):
            _cfg = bundles[_id].config
            try:
                _seed, _target, _pair_target, _ = get_seeds(_id)
                _scores, _final, _labels = ma.evaluate_seeds(
                    get_model(_id, parameter_noise.value), _seed, _target, _pair_target,
                    steps=eval_length.value * _cfg.run.t, horizon=_cfg.run.t, repeats=eval_repeats.value,
                    key=jr.fold_in(jr.PRNGKey(eval_seed.value), zlib.crc32(_id.encode())), tolerance=tolerance.value,
                    memory_gb=memory_budget.value,
                )
                # Final states are only drawn, so half precision is enough
                score_cache[_id, _settings] = {"rows": _scores, "final": _final.astype(np.float16), "labels": _labels}
            except (FileNotFoundError, ValueError) as _error:
                _failures.append(f"- `{_id}`: {_error}")

    _rows = []
    # model_id -> (final RGBA [pair, rollout, 4, H, W], outcome labels [pair, rollout])
    final_states = {}
    for _id in selected.model_id:
        if (_id, _settings) not in score_cache:
            continue
        _entry = score_cache[_id, _settings]
        _names = ma.target_names(bundles[_id].config)
        for _row in _entry["rows"]:
            _rows.append({"model_id": _id, "target_name": _names[_row["target"]], **_row})
        final_states[_id] = (_entry["final"], _entry["labels"])
    pair_scores = pd.DataFrame(_rows)
    _missing = len(selected) - (pair_scores.model_id.nunique() if len(pair_scores) else 0)
    mo.stop(
        not len(pair_scores),
        mo.callout(mo.md("Press **Evaluate selected models**.\n\n" + "\n".join(_failures)), kind="neutral"),
    )
    model_scores = selected.merge(
        pair_scores.groupby("model_id").agg(
            correct=("correct", "mean"),
            worst_seed_correct=("correct", "min"),
            wrong_target=("wrong_target", "mean"),
            other=("other", "mean"),
            failed=("failed", "mean"),
            mse_at_t=("mse_at_t", "mean"),
            mse_final=("mse_final", "mean"),
        ).reset_index(),
        on="model_id",
    )
    _notes = []
    if _missing:
        _notes.append(f"{_missing} selected models have no scores for these settings; press the button to add them.")
    if _failures:
        _notes.append("Failed:\n\n" + "\n".join(_failures))
    mo.callout(mo.md("\n\n".join(_notes)), kind="warn") if _notes else mo.md(
        f"Scores for **{len(model_scores)}** models."
    )
    return final_states, model_scores, pair_scores


@app.cell(hide_code=True)
def _(model_scores):
    _factors = ["attractors", "family", "padding", "stability_mode", "channels", "t", "targets", "wandb_group"]
    _factors = [_f for _f in _factors if _f in model_scores]
    _metrics = ["correct", "worst_seed_correct", "wrong_target", "other", "failed", "mse_at_t", "mse_final", "best_loss"]
    plot_x = mo.ui.dropdown(_factors, value="attractors", label="x")
    plot_hue = mo.ui.dropdown(["(none)"] + _factors, value="family", label="colour")
    plot_y = mo.ui.dropdown(_metrics, value="correct", label="y")
    mo.hstack([plot_x, plot_hue, plot_y], justify="start")
    return plot_hue, plot_x, plot_y


@app.cell(hide_code=True)
def _(model_scores, pair_scores, plot_hue, plot_x, plot_y):
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    strip_plot(_ax, model_scores, plot_x.value, plot_y.value, None if plot_hue.value == "(none)" else plot_hue.value)
    if plot_y.value.startswith("mse") or plot_y.value == "best_loss":
        _ax.set_yscale("log")
    _ax.set_title("one dot per model, bar = mean", fontsize=9)
    _fig.tight_layout()

    _by_target = pair_scores.merge(model_scores[["model_id", "attractors"]], on="model_id").pivot_table(
        index="target_name", columns="attractors", values="correct", aggfunc="mean")
    _fig2, _ax2 = plt.subplots(figsize=(1.2 + 0.9 * _by_target.shape[1], 0.5 + 0.45 * _by_target.shape[0]))
    _image = _ax2.imshow(_by_target.to_numpy(float), cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for (_i, _j), _value in np.ndenumerate(_by_target.to_numpy(float)):
        if np.isfinite(_value):
            _ax2.text(_j, _i, f"{_value:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if _value < 0.5 else "black")
    _ax2.set_xticks(range(_by_target.shape[1]), _by_target.columns)
    _ax2.set_yticks(range(_by_target.shape[0]), _by_target.index)
    _ax2.set_xlabel("attractors")
    _ax2.set_title("correct, by target", fontsize=9)
    _fig2.tight_layout()

    mo.vstack([
        mo.hstack([_fig, _fig2], justify="start", align="start"),
        mo.ui.table(model_scores.round(4), selection=None, page_size=10, label="Per model"),
        mo.accordion({"Per seed": mo.ui.table(pair_scores.round(4), selection=None, page_size=10)}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Final states

    - One panel per group (default: number of attractors), side by side
    - Rows are models, columns are their training seeds; the title is the target that seed should grow
    - Frame: green = grew its own target, red = anything else (the outcome is written underneath)
    """)
    return


@app.cell(hide_code=True)
def _(eval_repeats, model_scores):
    _factors = [_f for _f in ("attractors", "family", "padding", "stability_mode", "targets", "wandb_group")
                if _f in model_scores]
    gallery_group = mo.ui.dropdown(_factors, value="attractors", label="Group by")
    gallery_order = mo.ui.dropdown(["best first", "worst first", "table order"], value="best first", label="Order")
    gallery_count = mo.ui.number(1, 50, value=6, step=1, label="Models per group")
    gallery_rollout = mo.ui.number(0, eval_repeats.value - 1, value=0, step=1, label="Rollout shown")
    mo.hstack([gallery_group, gallery_order, gallery_count, gallery_rollout], justify="start", gap=1.5, wrap=True)
    return gallery_count, gallery_group, gallery_order, gallery_rollout


@app.cell(hide_code=True)
def _(
    bundles,
    final_states,
    gallery_count,
    gallery_group,
    gallery_order,
    gallery_rollout,
    model_scores,
):
    _scores = model_scores.copy()
    _scores[gallery_group.value] = _scores[gallery_group.value].where(_scores[gallery_group.value].notna(), "none")
    if gallery_order.value != "table order":
        _best_first = gallery_order.value == "best first"
        _scores = _scores.sort_values(["correct", "mse_final"], ascending=[not _best_first, _best_first])

    _panels = []
    for _level in levels(_scores[gallery_group.value]):
        _rows = _scores[_scores[gallery_group.value] == _level].head(gallery_count.value)
        _columns = max(len(final_states[_id][1]) for _id in _rows.model_id)
        _fig, _axes = plt.subplots(len(_rows), _columns, figsize=(1.15 * _columns + 0.9, 1.3 * len(_rows) + 0.4),
                                   squeeze=False)
        for _i, _row in enumerate(_rows.itertuples()):
            _final, _labels = final_states[_row.model_id]
            _cfg = bundles[_row.model_id].config
            _names = ma.target_names(_cfg)
            _pair_target = [_names.index(ma.emoji_name(_pair.target)) for _pair in _cfg.data.emoji.pairs]
            _rollout = min(gallery_rollout.value, _final.shape[1] - 1)
            for _j in range(_columns):
                _ax = _axes[_i, _j]
                if _j >= len(_final):
                    _ax.axis("off")
                    continue
                _label = int(_labels[_j, _rollout])
                _correct = _label == _pair_target[_j]
                show_image(_ax, _final[_j, _rollout].astype(np.float32), _names[_pair_target[_j]])
                for _spine in _ax.spines.values():
                    _spine.set_color("#1baf7a" if _correct else "#e34948")
                    _spine.set_linewidth(2.5)
                if not _correct:
                    _ax.set_xlabel(f"→ {ma.outcome_name(_label, _names)}", fontsize=7, color="#e34948", labelpad=2)
            _axes[_i, 0].set_ylabel(
                f"{_row.family} · {_row.padding}\nrep {_row.repeat} · {_row.model_id[-8:]}\ncorrect {_row.correct:.0%}",
                fontsize=7,
            )
        _fig.suptitle(f"{gallery_group.value} = {_level} ({len(_rows)} of "
                      f"{int((_scores[gallery_group.value] == _level).sum())} models)", fontsize=9)
        _fig.tight_layout()
        _panels.append(_fig)
    mo.hstack(_panels, justify="start", align="start", wrap=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Input space

    - The chosen colour (RGBA) is written into the seed's pixels; hidden channels come from the *hidden channels* seed
    - Two views of the same colour: the hue-saturation wheel (at the value set below it) and the RGB cube
    - Wheel: click or drag; cube: drag to rotate, click a wall; in both, a dot selects that training seed colour
    - The red/green/blue/alpha sliders reach any colour, including the inside of the cube
    - Pixel seeds (`mode: pixel`) differ only in their hidden code, so their dots overlap; choose the code with *Hidden channels*
    - Lines in the wheel and faces in the cube outline where a **stable attractor** is reached, as in
      `spatial_position_scan_combined.py`: the state is classified after every step (closest target,
      `INTERMEDIATE` when two targets each own ≥ 20 % of the live pixels, `NO_VISIBLE_PATTERN` when none is alive);
      a colour counts when it ends on a target (`reaches_pure_attractor`), optionally also only when the state
      has stopped changing (`converged_before_end`). The wheel's lines come from the hue × saturation basin map,
      the cube's faces from the basin cube (both below; press their buttons first)
    """)
    return


@app.cell(hide_code=True)
def _(candidates, selected):
    _pool = selected if len(selected) else candidates
    _options = {
        f"{_row.targets} · {_row.family} · {_row.padding} · {_row.stability_mode or ''} · rep {_row.repeat} · {_row.model_id[:8]}": _row.model_id
        for _row in _pool.itertuples()
    }
    explore_model = mo.ui.dropdown(_options, value=next(iter(_options), None), label="Model", full_width=True)
    explore_model
    return (explore_model,)


@app.cell(hide_code=True)
def _(bundles, explore_model, get_seeds):
    mo.stop(explore_model.value is None, mo.md("Select a model."))
    explore_id = explore_model.value
    explore_cfg = bundles[explore_id].config
    try:
        explore_seeds, explore_targets, explore_pair_target, _warning = get_seeds(explore_id)
    except FileNotFoundError as _error:
        mo.stop(True, mo.callout(f"Could not load the emoji images: {_error}", kind="danger"))
    explore_names = ma.target_names(explore_cfg)
    explore_footprint = ma.seed_footprint(explore_seeds)
    explore_threshold_base = ma.target_gap(explore_targets)
    seed_rgba = ma.seed_colours(explore_seeds, explore_footprint)

    _n = len(explore_seeds)
    _fig, _axes = plt.subplots(2, _n, figsize=(1.8 * _n, 3.6), squeeze=False)
    for _pair in range(_n):
        _target = explore_pair_target[_pair]
        _swatch = np.broadcast_to(seed_rgba[_pair][:, None, None], (4, 1, 1))
        show_image(_axes[0, _pair], _swatch, f"seed {_pair}: RGBA\n{np.round(seed_rgba[_pair], 2).tolist()}")
        show_image(_axes[1, _pair], explore_targets[_target], explore_names[_target], outcome_colour(_target))
    _fig.suptitle(f"{_n} training seeds ({int(explore_footprint.sum())} seed pixel(s)) and their targets", fontsize=9)
    _fig.tight_layout()
    _items = [_fig]
    if _warning:
        _items.insert(0, mo.callout(f"The reloaded seeds differ from training: {_warning}", kind="warn"))
    mo.vstack(_items)
    return (
        explore_cfg,
        explore_footprint,
        explore_id,
        explore_names,
        explore_pair_target,
        explore_seeds,
        explore_targets,
        explore_threshold_base,
        seed_rgba,
    )


@app.cell(hide_code=True)
def _(explore_names, explore_pair_target, explore_seeds, seed_rgba):
    seed_markers = [
        {"r": _r, "g": _g, "b": _b, "a": _a, "label": explore_names[int(explore_pair_target[_pair])],
         "colour": outcome_colour(int(explore_pair_target[_pair]))}
        for _pair, (_r, _g, _b, _a) in enumerate(np.clip(seed_rgba, 0.0, 1.0).astype(float).tolist())
    ]
    _first = seed_markers[0]
    picker = mo.ui.anywidget(RGBCube(r=_first["r"], g=_first["g"], b=_first["b"], a=_first["a"],
                                     markers=seed_markers, wheel=True))

    # Hidden channels only matter for pixel seeds that set them
    _has_hidden = bool(np.any(explore_seeds[:, 4:] != 0))
    _hidden_options = {"zeros": -1} | {
        f"seed {_pair} ({explore_names[explore_pair_target[_pair]]})": _pair for _pair in range(len(explore_seeds))
    }
    # Pixel seeds differ in their hidden code, so start from the first seed's code
    hidden_base = mo.ui.dropdown(_hidden_options, value=list(_hidden_options)[1 if _has_hidden else 0],
                                 label="Hidden channels")
    explore_length = mo.ui.slider(1, 8, value=3, step=1, label="Length (× run.t)", show_value=True)
    explore_repeats = mo.ui.number(1, 32, value=1, step=1, label="Rollouts")
    stable_rule = mo.ui.dropdown(
        {"ends on a target (reaches_pure_attractor)": False,
         "… and has settled (converged_before_end)": True},
        value="ends on a target (reaches_pure_attractor)", label="Stable attractor",
    )
    show_boundaries = mo.ui.checkbox(value=True, label="Draw stable-attractor boundaries")
    field_mode = mo.ui.radio(
        {"colours": "colours", "steps to stable attractor (normal system)": "normal",
         "steps to stable image (dynamic system)": "dynamic"},
        value="colours", label="Wheel and cube show",
    )
    _controls = [explore_length, explore_repeats] + ([hidden_base] if _has_hidden else []) + [
        stable_rule, show_boundaries, field_mode]
    mo.hstack([picker, mo.vstack(_controls)], justify="start", align="start", gap=2)
    return (
        explore_length,
        explore_repeats,
        field_mode,
        hidden_base,
        picker,
        seed_markers,
        show_boundaries,
        stable_rule,
    )


@app.cell(hide_code=True)
def _(explore_seeds, hidden_base):
    # The seed the chosen colours are written into
    base_seed = np.zeros_like(explore_seeds[0]) if hidden_base.value < 0 else explore_seeds[hidden_base.value].copy()
    base_seed[:4] = 0.0
    return (base_seed,)


@app.cell(hide_code=True)
def _(
    base_seed,
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_names,
    explore_repeats,
    explore_targets,
    explore_threshold_base,
    get_model,
    memory_budget,
    parameter_noise,
    picker,
    stable_rule,
    tolerance,
):
    _rgba = np.array([[picker.value[_name] for _name in ("r", "g", "b", "a")]], dtype=np.float32)
    _states = np.repeat(ma.coloured_seeds(base_seed, explore_footprint, _rgba), explore_repeats.value, axis=0)
    _steps = explore_length.value * explore_cfg.run.t
    _n_frames = math.gcd(_steps, 8)
    _distances, _trajectory, _classes, _changes = ma.rollout(
        get_model(explore_id, parameter_noise.value), _states, explore_targets, _steps, jr.PRNGKey(eval_seed.value),
        frames=_n_frames, memory_gb=memory_budget.value, dynamics=True,
    )
    _threshold = tolerance.value * explore_threshold_base
    _labels = ma.classify(_distances[:, -1], _trajectory[:, -1], _threshold)
    _dynamics = ma.attractor_dynamics(_classes, _changes, len(explore_names))
    _stable = ma.stable_attractor(_dynamics, settled=stable_rule.value)

    def _stable_text(_r):
        if _stable[_r] >= 0:
            return f"stable {explore_names[_stable[_r]]} from step {_dynamics['steps_to_attractor'][_r]}"
        _final = ma.scan_class_name(_dynamics["final_class"][_r], explore_names)
        return f"not stable ({_final}{', unsettled' if _dynamics['reaches_attractor'][_r] else ''})"

    _frames = np.arange(_n_frames + 1) * (_steps // _n_frames)
    _columns = max(len(_frames), explore_repeats.value)
    _fig = plt.figure(figsize=(1.5 * _columns, 6.4))
    _grid = _fig.add_gridspec(3, _columns, height_ratios=[1, 1, 1.6])
    for _i, _frame in enumerate(_frames):
        show_image(_fig.add_subplot(_grid[0, _i]), _trajectory[0, _i], f"step {_frame}")
    for _r in range(explore_repeats.value):
        _label = int(_labels[_r])
        show_image(_fig.add_subplot(_grid[1, _r]), _trajectory[_r, -1],
                   f"rollout {_r}: {ma.outcome_name(_label, explore_names)}\n{_stable_text(_r)}",
                   outcome_colour(_label))
    _ax = _fig.add_subplot(_grid[2, :])
    for _k, _name in enumerate(explore_names):
        _ax.plot(_distances[:, :, _k].T, color=outcome_colour(_k), lw=1, alpha=0.6)
        _ax.plot([], [], color=outcome_colour(_k), label=_name)
    _ax.axhline(_threshold, color="#555", ls="--", lw=1, label="tolerance")
    _ax.axvline(explore_cfg.run.t, color="#999", ls=":", lw=1, label="run.t")
    _ax.set_yscale("log")
    _ax.set_xlabel("step")
    _ax.set_ylabel("RGBA MSE to target")
    _ax.legend(frameon=False, fontsize=8, ncol=len(explore_names) + 2)
    style_axes(_ax)
    _fig.suptitle(f"seed RGBA {np.round(_rgba[0], 3).tolist()}: rollout 0 (top), end of every rollout (middle)",
                  fontsize=9)
    _fig.tight_layout()
    # Distances of every rollout of the chosen colour after every step, for the velocity map
    chosen_distances = _distances
    _fig
    return (chosen_distances,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Velocity map

    - Every state is placed by its similarity to each target: one axis per target, from 0 (as far from that
      target as an empty grid) to 1 (identical to the target image, shown at the end of its axis).
      Similarity = 1 − RGBA MSE to the target / RGBA MSE of an empty grid to it
    - Stars: where the target images themselves lie. Each path starts (circle) at its start signal, i.e. the seed
      used for the rollout, and ends (square) after *Length* × `run.t` steps; *Rollouts* paths per start
    - Arrows: each goes from the state at step t to the state at step t + *arrow spacing*, so they chain along the
      path and a long arrow means fast. The graph below shows the similarity to the end attractor (the target
      closest at the end) and its velocity per step
    - 2 targets: one plane; 3 targets: a 3D view and its three planes; more: the plane of every pair of targets
    - The thin line is the path at every step, the arrows only every *arrow spacing* steps. Where the state
      hardly changes any more (at the end, close to its attractor) the arrows become shorter than their heads
      and vanish behind the line: there the run is (nearly) at rest
    - *Axes*: the planes also with start axes. Each axis starts at the lowest start point (bottom left); half of
      its length is logarithmic in the offset d from there, log10(1 + d / c) with c = spread of the start points /
      *magnify*, the other half linear in d. Near the start points (which differ only by about 1e-4) it is
      logarithmic, so their positions relative to each other are visible (ticks: + offset from the lowest start
      point); further out it becomes linear, so the course of the paths keeps its shape (ticks: the similarity)
    - *Colours between the trained seeds*: *per pair of seeds* start points evenly between every two training
      seeds (the whole seeds are mixed), one rollout each with the same random keys, drawn thin in their start
      colour (pixel seeds that only differ in their hidden code: a blend of their targets' colours). They are not
      in the graph below
    """)
    return


@app.cell(hide_code=True)
def _(explore_names, explore_pair_target, explore_seeds):
    _options = {f"seed {_pair} → {explore_names[int(explore_pair_target[_pair])]}": _pair
                for _pair in range(len(explore_seeds))}
    _options["chosen colour"] = -1
    _options["specific start point"] = -2
    velocity_starts = mo.ui.multiselect(_options, value=list(_options)[:-1], label="Start signals")
    velocity_stride = mo.ui.slider(1, 32, value=8, step=1, label="Arrow spacing (steps)", show_value=True)
    velocity_smooth = mo.ui.slider(1, 32, value=8, step=1, label="Velocity smoothing (steps)", show_value=True)
    velocity_background = mo.ui.checkbox(value=False, label="Start-point basins as background")
    velocity_axes = mo.ui.radio({"linear": "linear", "logarithmic": "log", "both": "both"}, value="both",
                                label="Axes", inline=True)
    velocity_magnify = mo.ui.dropdown({"10×": 10.0, "50×": 50.0, "200×": 200.0, "1000×": 1000.0}, value="50×",
                                      label="Log axes: magnify the start points")
    velocity_between = mo.ui.checkbox(value=False, label="Colours between the trained seeds")
    velocity_between_n = mo.ui.number(1, 100, value=20, step=1, label="per pair of seeds")
    mo.vstack([
        mo.hstack([velocity_starts, velocity_stride, velocity_smooth, velocity_background], justify="start",
                  gap=1.5, wrap=True),
        mo.hstack([velocity_axes, velocity_magnify, velocity_between, velocity_between_n], justify="start",
                  gap=1.5, wrap=True),
    ])
    return (
        velocity_axes,
        velocity_background,
        velocity_between,
        velocity_between_n,
        velocity_magnify,
        velocity_smooth,
        velocity_starts,
        velocity_stride,
    )


@app.cell(hide_code=True)
def _(
    eval_seed,
    explore_cfg,
    explore_id,
    explore_length,
    explore_repeats,
    explore_seeds,
    explore_targets,
    get_model,
    memory_budget,
    parameter_noise,
):
    # Every training seed (with its hidden channels, as in training), rolled out like the chosen colour
    _distances, _ = ma.rollout(
        get_model(explore_id, parameter_noise.value), np.repeat(explore_seeds, explore_repeats.value, axis=0),
        explore_targets, explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
        memory_gb=memory_budget.value,
    )
    # [pair, rollout, step, target]
    seed_distances = _distances.reshape(len(explore_seeds), explore_repeats.value, *_distances.shape[1:])
    return (seed_distances,)


@app.cell(hide_code=True)
def _(
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_names,
    explore_pair_target,
    explore_seeds,
    explore_targets,
    get_model,
    memory_budget,
    parameter_noise,
    velocity_between,
    velocity_between_n,
):
    # N start points evenly between every two training seeds, t = 1 / (N + 1) … N / (N + 1). The whole
    # seeds are mixed (colour and, for pixel seeds, the hidden code); one rollout each, with the same
    # random keys for all, so they differ only by their start
    between_paths = []
    if velocity_between.value:
        _t = np.arange(1, velocity_between_n.value + 1) / (velocity_between_n.value + 1)
        _meta = [(_a, _b, _x) for _a, _b in itertools.combinations(range(len(explore_seeds)), 2) for _x in _t]
        _states = np.array([(1 - _x) * explore_seeds[_a] + _x * explore_seeds[_b] for _a, _b, _x in _meta])
        _distances, _ = ma.rollout(
            get_model(explore_id, parameter_noise.value), _states, explore_targets,
            explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value), memory_gb=memory_budget.value,
            shared_key=True,
        )
        _colours = np.clip(ma.seed_colours(_states, explore_footprint)[:, :3], 0.0, 1.0)
        _seed_colours = ma.seed_colours(explore_seeds, explore_footprint)[:, :3]
        for (_a, _b, _x), _d, _c in zip(_meta, _distances, _colours):
            # The start colour itself; pixel seeds that differ only in their hidden code all look the same,
            # so they get a blend of their targets' colours instead
            if np.abs(_seed_colours[_a] - _seed_colours[_b]).max() < 0.05:
                _c = (1 - _x) * np.array(to_rgb(outcome_colour(int(explore_pair_target[_a])))) \
                    + _x * np.array(to_rgb(outcome_colour(int(explore_pair_target[_b]))))
            between_paths.append({
                "seeds": (_a, _b), "t": float(_x), "colour": to_hex(_c), "distances": _d[None],
                "label": f"seed {_a} ({explore_names[int(explore_pair_target[_a])]}) → seed {_b} "
                         f"({explore_names[int(explore_pair_target[_b])]})",
            })
    return (between_paths,)


@app.cell(hide_code=True)
def _(
    between_paths,
    chosen_distances,
    explore_cfg,
    explore_names,
    explore_pair_target,
    explore_targets,
    seed_distances,
    special_distances,
    start_planes,
    velocity_axes,
    velocity_background,
    velocity_between_n,
    velocity_magnify,
    velocity_smooth,
    velocity_starts,
    velocity_stride,
):
    _paths = []
    for _start in velocity_starts.value:
        if _start == -1:
            _label, _colour, _distances = "chosen colour", "black", chosen_distances
        elif _start == -2:
            _label, _colour, _distances = "specific start point", SPECIAL_COLOUR, special_distances
        else:
            _label = f"seed {_start} → {explore_names[int(explore_pair_target[_start])]}"
            _colour, _distances = outcome_colour(int(explore_pair_target[_start])), seed_distances[_start]
        _paths.append({"label": _label, "colour": _colour, "style": LINESTYLES[len(_paths) % len(LINESTYLES)],
                       "sim": ma.similarity(_distances, explore_targets)})
    _main_paths = list(_paths)
    # The start colours between the trained seeds: thin, below the others, one legend entry per pair of seeds
    _between = [{"label": None, "colour": _p["colour"], "style": "-", "small": True,
                 "sim": ma.similarity(_p["distances"], explore_targets)} for _p in between_paths]
    _between_legend = []
    for _seeds in dict.fromkeys(_p["seeds"] for _p in between_paths):
        _group = [_p for _p in between_paths if _p["seeds"] == _seeds]
        _between_legend.append(Line2D([], [], color=_group[len(_group) // 2]["colour"], lw=1.5,
                                      label=f"{velocity_between_n.value} colours,\n{_group[0]['label']}"))
    _paths = _between + _paths
    mo.stop(not _paths, mo.md("Choose at least one start signal."))

    _n = len(explore_names)
    _target_sim = ma.target_similarity(explore_targets)
    _background = start_planes if velocity_background.value else None
    _limits = similarity_limits(_paths + [{"sim": _plane["sim"]} for _plane in _background or []], _target_sim)
    # Start axes for the log view: range of everything drawn on each axis (no padding below, so the start
    # points sit at the bottom left unless something goes below them), anchored at the lowest start point
    _values = np.concatenate([_target_sim] + [_p["sim"].reshape(-1, _n) for _p in _paths]
                             + [_plane["sim"].reshape(-1, _n) for _plane in _background or []])
    _low = np.nanmin(np.where(np.isfinite(_values), _values, np.nan), axis=0)
    _high = np.maximum(np.nanmax(np.where(np.isfinite(_values), _values, np.nan), axis=0), 1.0)
    _log = start_axes(_paths, velocity_magnify.value, _low, _high)
    _log_limits = {_k: (float(_low[_k]), float(_high[_k])) for _k in range(_n)}
    _stride = velocity_stride.value
    _planes = list(itertools.combinations(range(_n), 2))
    _legend = similarity_legend(_paths) + _between_legend + (start_basin_legend(explore_names) if _background else [])
    _figures = []
    if velocity_background.value and start_planes is None:
        _figures.append(mo.callout("For the background, compute the start-point basins below.", kind="neutral"))
    if _n == 3:
        _fig3d = plt.figure(figsize=(8.0, 5.8))
        similarity_space_3d(_fig3d.add_subplot(projection="3d"), _paths, _target_sim, explore_names, _stride,
                            _limits)
        _fig3d.subplots_adjust(left=0.0, right=0.72, bottom=0.02, top=0.98)
        _fig3d.legend(handles=similarity_legend(_paths) + _between_legend, loc="center left",
                      bbox_to_anchor=(0.74, 0.5), frameon=False, fontsize=8)
        _figures.append(_fig3d)
    # The planes, with linear and/or symmetric-logarithmic axes
    for _scale in ("linear", "log"):
        if velocity_axes.value not in (_scale, "both"):
            continue
        _columns = min(3, len(_planes))
        _rows = math.ceil(len(_planes) / _columns)
        _width = 4.8 * _columns + 2.8
        _fig, _axes = plt.subplots(_rows, _columns, figsize=(_width, 4.8 * _rows + 0.6), squeeze=False)
        for _k, (_ax, (_i, _j)) in enumerate(zip(_axes.flat, _planes)):
            if _background:
                start_basin_background(_ax, _background[_k], _i, _j, _n, log=_log if _scale == "log" else None)
            similarity_plane(_ax, _i, _j, _paths, _target_sim, explore_targets, explore_names, _stride,
                             _log_limits if _scale == "log" else _limits, log=_log if _scale == "log" else None)
        for _ax in _axes.flat[len(_planes):]:
            _ax.axis("off")
        # The legend in a margin on the right
        _fig.tight_layout(rect=(0, 0, 1 - 2.8 / _width, 0.93), w_pad=4, h_pad=3)
        _fig.legend(handles=_legend, loc="center left", bbox_to_anchor=(1 - 2.7 / _width, 0.5), frameon=False,
                    fontsize=8)
        _fig.suptitle("linear axes" if _scale == "linear" else
                      "start points at the bottom left: near them logarithmic (+ offsets from the lowest start "
                      "point),\nfurther out linear, so the paths keep their shape; the arrows join the same steps "
                      "as in the linear plot", fontsize=9)
        _figures.append(_fig)

    # Similarity to the end attractor (the closest target at the end) and its velocity per step
    _fig_v, (_ax_s, _ax_v) = plt.subplots(1, 2, figsize=(12, 3.8))
    for _path in _main_paths:
        for _r, _sim in enumerate(_path["sim"]):
            _end = int(np.nanargmax(_sim[-1])) if np.isfinite(_sim[-1]).any() else 0
            _towards = _sim[:, _end]
            _velocity = ma.moving_average(np.diff(_towards), velocity_smooth.value)
            _label = f"{_path['label']} (ends: {explore_names[_end]})" if _r == 0 else None
            _ax_s.plot(np.arange(len(_towards)), _towards, color=_path["colour"], ls=_path["style"], lw=1.2,
                       label=_label)
            _ax_v.plot(np.arange(1, len(_towards)), _velocity, color=_path["colour"], ls=_path["style"], lw=1.2)
    _ax_s.axhline(1.0, color="#999", ls="--", lw=0.8)
    _ax_s.set_ylabel("similarity to end attractor")
    _ax_v.axhline(0.0, color="#999", lw=0.8)
    _ax_v.set_ylabel("velocity towards end attractor\n(Δ similarity per step)")
    for _ax in (_ax_s, _ax_v):
        _ax.axvline(explore_cfg.run.t, color="#999", ls=":", lw=1)
        _ax.set_xlabel("step")
        style_axes(_ax)
    _ax_s.legend(frameon=False, fontsize=8)
    _ax_v.set_title(f"moving average over {velocity_smooth.value} steps; dotted: run.t", fontsize=9)
    _fig_v.tight_layout()
    mo.vstack([mo.hstack(_figures, justify="start", align="start", wrap=True), _fig_v])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA of the states

    - The same runs as in the velocity map (its *start signals*, *Rollouts*, *Length* and, when ticked, the colours
      between the trained seeds), but placed by their whole state instead of the similarity to the targets: a PCA
      of the states (every channel at every pixel, or RGBA only) kept every *keep every* steps
    - The axes are the directions in which these states differ most (PC1, PC2, …, with their share of the
      variance). They are not fixed in advance, so states that resemble neither target, and differences only in the
      hidden channels (e.g. pixel seeds with different hidden codes), show up as well
    - *Standardise*: every channel is divided by its spread, so channels with large values do not dominate
    - *Fit on*: the PCA is computed from the whole paths, only their *first … steps*, or only the start states, and
      then all states are placed with it. Fitted on the first steps (or the starts), the axes show how the start
      points and the first moves differ; fitted on the whole paths, the big moves to the attractors dominate
    - The PCA is a rotation of the state space, so distances and arrow lengths are distances between states (in
      standardised units): arrows join kept states about *arrow spacing* steps apart. Target images (stars) only
      with RGBA, since they have no hidden channels. *Axes range: start points* fits the axes to the start points
    - Below: share of the variance per component, and what the first components are (where in the grid, and which
      channels)
    """)
    return


@app.cell(hide_code=True)
def _():
    pca_channels = mo.ui.radio({"all channels": "all", "RGBA only": "rgba"}, value="all channels", label="Channels",
                               inline=True)
    pca_standardize = mo.ui.checkbox(value=True, label="Standardise each channel")
    pca_fit = mo.ui.radio({"whole paths": "all", "first steps": "first", "start states": "start"},
                          value="whole paths", label="Fit on", inline=True)
    pca_first = mo.ui.number(1, 8192, value=16, step=1, label="first … steps")
    pca_every = mo.ui.number(1, 64, value=4, step=1, label="Keep every … steps")
    pca_view = mo.ui.radio({"whole paths": "all", "start points": "start"}, value="whole paths", label="Axes range",
                           inline=True)
    pca_run = mo.ui.run_button(label="Compute PCA")
    mo.vstack([
        mo.hstack([pca_channels, pca_standardize, pca_every, pca_run], justify="start", gap=1.5, wrap=True),
        mo.hstack([pca_fit, pca_first, pca_view], justify="start", gap=1.5, wrap=True),
    ])
    return pca_channels, pca_every, pca_first, pca_fit, pca_run, pca_standardize, pca_view


@app.cell
def _():
    # PCAs computed in this session, keyed by their settings
    pca_cache = {}
    return (pca_cache,)


@app.cell(hide_code=True)
def _(
    base_seed,
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_names,
    explore_pair_target,
    explore_repeats,
    explore_seeds,
    explore_targets,
    get_model,
    hidden_base,
    memory_budget,
    parameter_noise,
    pca_cache,
    pca_channels,
    pca_every,
    pca_first,
    pca_fit,
    pca_run,
    pca_standardize,
    picker,
    start_weights,
    velocity_between,
    velocity_between_n,
    velocity_starts,
):
    _steps = explore_length.value * explore_cfg.run.t
    # Keep every … steps must divide the run length
    _every = math.gcd(_steps, int(pca_every.value))
    _chosen = np.array([[picker.value[_name] for _name in ("r", "g", "b", "a")]], dtype=np.float32)
    # The chosen colour and the specific start point only matter when they are start signals
    _key = (explore_id, parameter_noise.value, explore_length.value, hidden_base.value, eval_seed.value,
            explore_repeats.value, tuple(velocity_starts.value), velocity_between.value, velocity_between_n.value,
            tuple(np.round(_chosen[0], 4)) if -1 in velocity_starts.value else None,
            tuple(start_weights.value) if -2 in velocity_starts.value else None,
            pca_channels.value, pca_standardize.value, pca_fit.value, pca_first.value, _every)
    if pca_run.value and _key not in pca_cache and (velocity_starts.value or velocity_between.value):
        _model = get_model(explore_id, parameter_noise.value)
        _repeats = explore_repeats.value

        def _run(states, shared_key=False):
            return ma.rollout_states(_model, states, _steps, jr.PRNGKey(eval_seed.value), every=_every,
                                     memory_gb=memory_budget.value, shared_key=shared_key)

        # The same batches and keys as in the velocity map, so the runs are the same
        _paths, _runs = [], []
        with mo.status.spinner("Rolling out the start signals and keeping their states"):
            _seed_ids = [_s for _s in velocity_starts.value if _s >= 0]
            _all_seeds = _run(np.repeat(explore_seeds, _repeats, axis=0)) if _seed_ids else None
            for _start in velocity_starts.value:
                if _start == -1:
                    _label, _colour = "chosen colour", "black"
                    _states = _run(np.repeat(ma.coloured_seeds(base_seed, explore_footprint, _chosen), _repeats, axis=0))
                elif _start == -2:
                    _label, _colour = "specific start point", SPECIAL_COLOUR
                    _states = _run(np.repeat(ma.mixed_states(base_seed, explore_targets, [start_weights.value]),
                                             _repeats, axis=0))
                else:
                    _label = f"seed {_start} → {explore_names[int(explore_pair_target[_start])]}"
                    _colour = outcome_colour(int(explore_pair_target[_start]))
                    _states = _all_seeds[_start * _repeats:(_start + 1) * _repeats]
                _paths.append({"label": _label, "colour": _colour, "style": LINESTYLES[len(_paths) % len(LINESTYLES)]})
                _runs.append(_states)
            if velocity_between.value:
                # As in the velocity map: N seeds evenly between every two training seeds, same keys for all
                _t = np.arange(1, velocity_between_n.value + 1) / (velocity_between_n.value + 1)
                _meta = [(_a, _b, _x) for _a, _b in itertools.combinations(range(len(explore_seeds)), 2) for _x in _t]
                _mixed = np.array([(1 - _x) * explore_seeds[_a] + _x * explore_seeds[_b] for _a, _b, _x in _meta])
                _between = _run(_mixed, shared_key=True)
                _colours = np.clip(ma.seed_colours(_mixed, explore_footprint)[:, :3], 0.0, 1.0)
                _seed_colours = ma.seed_colours(explore_seeds, explore_footprint)[:, :3]
                for (_a, _b, _x), _states, _c in zip(_meta, _between, _colours):
                    if np.abs(_seed_colours[_a] - _seed_colours[_b]).max() < 0.05:
                        _c = (1 - _x) * np.array(to_rgb(outcome_colour(int(explore_pair_target[_a])))) \
                            + _x * np.array(to_rgb(outcome_colour(int(explore_pair_target[_b]))))
                    _paths.insert(0, {"label": None, "colour": to_hex(_c), "style": "-", "small": True})
                    _runs.insert(0, _states[None])
        _frames = _runs[0].shape[1]
        _all = np.concatenate([_r.reshape(-1, *_r.shape[2:]) for _r in _runs])
        _step = np.tile(np.arange(_frames) * _every, len(_all) // _frames)
        _fit = {"all": np.ones(len(_all), bool), "first": _step <= pca_first.value, "start": _step == 0}[pca_fit.value]
        with mo.status.spinner(f"PCA of {len(_all)} states ({_fit.sum()} fitted)"):
            _pca = ma.state_pca(_all, _fit, channels=None if pca_channels.value == "all" else [0, 1, 2, 3],
                                standardize=pca_standardize.value, n_components=10)
        _offset = 0
        for _path, _r in zip(_paths, _runs):
            _path["pcs"] = _pca["scores"][_offset:_offset + _r.shape[0] * _frames].reshape(_r.shape[0], _frames, -1)
            _offset += _r.shape[0] * _frames
        pca_cache[_key] = {
            "paths": _paths, "explained": _pca["explained"], "components": _pca["components"],
            "channels": _pca["channels"], "n_states": len(_all), "n_fit": _pca["n_fit"],
            "features": int(np.prod(_pca["components"].shape[1:])), "every": _every,
            "targets": ma.pca_project(_pca, explore_targets) if pca_channels.value == "rgba" else None,
        }
        del _all, _runs
    pca_result = pca_cache.get(_key)
    mo.md("Press **Compute PCA** (uses the start signals of the velocity map and the length, hidden channels, noise "
          "and memory settings above; the states are rolled out again and kept every "
          f"{_every} steps).") if pca_result is None else None
    return (pca_result,)


@app.cell(hide_code=True)
def _(explore_names, pca_result, pca_view, velocity_stride):
    mo.stop(pca_result is None)
    mo.stop(len(pca_result["explained"]) < 2,
            mo.md("The fitted states vary in fewer than two directions; fit on more states."))
    _paths = pca_result["paths"]
    _k = len(pca_result["explained"])
    _planes = [(0, 1), (0, 2), (1, 2)] if _k >= 3 else [(0, 1)]
    _stride = max(1, round(velocity_stride.value / pca_result["every"]))
    _zoom = pca_view.value == "start"
    _limits = pca_limits(_paths, pca_result["targets"], pca_view.value)
    _width = 4.6 * len(_planes) + 2.8
    _fig, _axes = plt.subplots(1, len(_planes), figsize=(_width, 5.0), squeeze=False)
    for _ax, (_a, _b) in zip(_axes.flat, _planes):
        pca_plane(_ax, _a, _b, _paths, pca_result["explained"], _stride, _limits, targets=pca_result["targets"],
                  names=explore_names, zoom=_zoom)
    _fig.tight_layout(rect=(0, 0, 1 - 2.8 / _width, 0.93), w_pad=3)
    _between = [_p for _p in _paths if _p.get("small")]
    _legend = similarity_legend(_paths)
    if _between:
        _legend.append(Line2D([], [], color=_between[len(_between) // 2]["colour"], lw=1.5,
                              label=f"{len(_between)} colours between the trained seeds"))
    if pca_result["targets"] is None:
        _legend = [_h for _h in _legend if _h.get_label() != "target image"]
    _fig.legend(handles=_legend, loc="center left", bbox_to_anchor=(1 - 2.7 / _width, 0.5), frameon=False, fontsize=8)
    _fig.suptitle(("start points (axes fitted to them, x and y scaled separately)" if _zoom else "whole paths")
                  + f"; arrows every {_stride * pca_result['every']} steps", fontsize=9)

    _fig_variance, _ax = plt.subplots(figsize=(4.2, 3.0))
    _ax.bar(np.arange(1, _k + 1), pca_result["explained"], color="#2a78d6")
    _ax.plot(np.arange(1, _k + 1), np.cumsum(pca_result["explained"]), color=INK_SECONDARY, marker="o", ms=3,
             label="cumulative")
    _ax.set_xticks(np.arange(1, _k + 1), [f"PC{_c}" for _c in range(1, _k + 1)], fontsize=7)
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("share of the variance")
    _ax.legend(frameon=False, fontsize=8)
    style_axes(_ax)
    _fig_variance.tight_layout()

    _channels = "all channels" if len(pca_result["channels"]) > 4 else "RGBA"
    mo.vstack([
        mo.md(f"{pca_result['n_states']} states ({pca_result['n_fit']} of them fitted), {pca_result['features']} "
              f"numbers each ({_channels}); PC1–PC{min(3, _k)} explain "
              f"{np.sum(pca_result['explained'][:3]):.0%} of the variance of the fitted states."),
        _fig,
        mo.hstack([_fig_variance, pca_components_figure(pca_result)], justify="start", align="start", wrap=True),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Start-point basins

    - Which start points in the similarity space lead to which stable attractor (criterion above)?
    - Start points are mixtures of two target images and the empty grid: weight *a* of the x target plus *b* of
      the y target with *a* + *b* ≤ 1, so they fill the area between the empty grid (0, 0) and the two target images
      (*grid steps* along the total amount and along the share of the y target; hidden channels from
      *Hidden channels*). Each is rolled out once for *Length* × `run.t` steps and placed at its own similarity;
      the shaded region and its outline show the attractor it reaches, grey = none
    - One plane per pair of targets, as in the velocity map; other targets have weight 0 there
    - The *specific start point* mixes all targets with the weights below; its path is drawn on top, and it can be
      added to the velocity map as a start signal. The velocity map can show these regions as its background
    """)
    return


@app.cell(hide_code=True)
def _(explore_names):
    start_steps = mo.ui.number(3, 41, value=15, step=1, label="Grid steps per target")
    start_run = mo.ui.run_button(label="Compute start-point basins")
    start_weights = mo.ui.array(
        [mo.ui.slider(0.0, 1.0, value=0.3, step=0.05, label=f"weight of {_name}", show_value=True)
         for _name in explore_names],
        label="Specific start point",
    )
    mo.hstack([mo.vstack([start_steps, start_run]), start_weights], justify="start", align="start", gap=2)
    return start_run, start_steps, start_weights


@app.cell
def _():
    # Start-point basins computed in this session, keyed by their settings
    start_cache = {}
    return (start_cache,)


@app.cell(hide_code=True)
def _(
    base_seed,
    eval_seed,
    explore_cfg,
    explore_id,
    explore_length,
    explore_targets,
    get_model,
    hidden_base,
    memory_budget,
    parameter_noise,
    start_cache,
    start_run,
    start_steps,
):
    _key = (explore_id, parameter_noise.value, start_steps.value, explore_length.value, hidden_base.value,
            eval_seed.value)
    _pairs = list(itertools.combinations(range(len(explore_targets)), 2))
    if start_run.value and _key not in start_cache:
        # Amount u of target image in total and share v of it from the y target:
        # weights a = u (1 - v), b = u v, so a + b <= 1 and the rest is empty grid
        _grid = np.linspace(0.0, 1.0, start_steps.value)
        _u, _v = np.meshgrid(_grid, _grid, indexing="ij")
        _weights = []
        for _i, _j in _pairs:
            _w = np.zeros((_u.size, len(explore_targets)), np.float32)
            _w[:, _i], _w[:, _j] = (_u * (1 - _v)).ravel(), (_u * _v).ravel()
            _weights.append(_w)
        _states = ma.mixed_states(base_seed, explore_targets, np.concatenate(_weights))
        with mo.status.spinner(f"Rolling out {len(_states)} start points"):
            _distances, _, _classes, _changes = ma.rollout(
                get_model(explore_id, parameter_noise.value), _states, explore_targets,
                explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
                memory_gb=memory_budget.value, dynamics=True,
            )
        start_cache[_key] = {
            "pairs": _pairs, "shape": _u.shape,
            # Similarity of each start point, [start, target]
            "sim": ma.similarity(_distances[:, 0], explore_targets),
            "dynamics": ma.attractor_dynamics(_classes, _changes, len(explore_targets)),
        }
    # None until computed for these settings
    start_basin = start_cache.get(_key)
    mo.md(f"Press **Compute start-point basins** ({len(_pairs) * start_steps.value ** 2} rollouts; uses the "
          "length, hidden channels, noise and memory settings above).") if start_basin is None else None
    return (start_basin,)


@app.cell(hide_code=True)
def _(start_basin, stable_rule):
    # One plane per pair of targets: similarity [a, b, target] and stable attractor [a, b] of each start point
    start_planes = None
    if start_basin is not None:
        _stable = ma.stable_attractor(start_basin["dynamics"], settled=stable_rule.value)
        _size = int(np.prod(start_basin["shape"]))
        start_planes = [
            {"pair": _pair,
             "sim": start_basin["sim"][_k * _size:(_k + 1) * _size].reshape(*start_basin["shape"], -1),
             "stable": _stable[_k * _size:(_k + 1) * _size].reshape(start_basin["shape"])}
            for _k, _pair in enumerate(start_basin["pairs"])
        ]
    return (start_planes,)


@app.cell(hide_code=True)
def _(
    base_seed,
    eval_seed,
    explore_cfg,
    explore_id,
    explore_length,
    explore_repeats,
    explore_targets,
    get_model,
    memory_budget,
    parameter_noise,
    start_weights,
):
    # The specific start point, rolled out like the chosen colour: distances [rollout, step, target],
    # RGBA at the start and the end [rollout, 2, 4, H, W] and the stable-attractor measures of every rollout
    _state = ma.mixed_states(base_seed, explore_targets, [start_weights.value])
    special_distances, special_images, _classes, _changes = ma.rollout(
        get_model(explore_id, parameter_noise.value), np.repeat(_state, explore_repeats.value, axis=0),
        explore_targets, explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
        memory_gb=memory_budget.value, dynamics=True,
    )
    special_dynamics = ma.attractor_dynamics(_classes, _changes, len(explore_targets))
    return special_distances, special_dynamics, special_images


@app.cell(hide_code=True)
def _(
    explore_cfg,
    explore_names,
    explore_targets,
    special_distances,
    special_dynamics,
    special_images,
    stable_rule,
    start_weights,
):
    # What the specific start point looks like at the start and at the end of every rollout
    _n = len(explore_names)
    _stable = ma.stable_attractor(special_dynamics, settled=stable_rule.value)
    _sim = ma.similarity(special_distances, explore_targets)
    _columns = 1 + len(special_images)
    _fig, _axes = plt.subplots(1, _columns, figsize=(2.3 * _columns + 0.6, 2.9), squeeze=False)

    def _similarities(values):
        return "\n".join(f"{_name} {_value:+.3f}" for _name, _value in zip(explore_names, values))

    show_image(_axes[0, 0], special_images[0, 0], f"start (step 0)\n{_similarities(_sim[0, 0])}", SPECIAL_COLOUR)
    for _r in range(len(special_images)):
        _final = ma.scan_class_name(special_dynamics["final_class"][_r], explore_names)
        _result = (f"stable {explore_names[_stable[_r]]} from step {special_dynamics['steps_to_attractor'][_r]}"
                   if _stable[_r] >= 0 else f"not stable: {_final}")
        show_image(_axes[0, 1 + _r], special_images[_r, -1],
                   f"end (step {_sim.shape[1] - 1}), rollout {_r}\n{_result}\n{_similarities(_sim[_r, -1])}",
                   outcome_colour(int(_stable[_r])) if _stable[_r] >= 0 else INK_SECONDARY)
    _weights = ", ".join(f"{_name} {_w:.2f}" for _name, _w in zip(explore_names, start_weights.value))
    _fig.suptitle(f"specific start point ({_weights}); similarities below each image", fontsize=9)
    _fig.tight_layout()
    special_figure = _fig
    return (special_figure,)


@app.cell(hide_code=True)
def _(
    explore_names,
    explore_targets,
    special_distances,
    special_figure,
    start_planes,
    start_weights,
    velocity_stride,
):
    mo.stop(start_planes is None, mo.vstack([
        special_figure, mo.md("Press **Compute start-point basins** above for the regions.")]))
    _n = len(explore_names)
    _special = {"label": "specific start point", "colour": SPECIAL_COLOUR, "style": "-",
                "sim": ma.similarity(special_distances, explore_targets)}
    _target_sim = ma.target_similarity(explore_targets)
    _limits = similarity_limits([_special] + [{"sim": _plane["sim"]} for _plane in start_planes], _target_sim)
    _columns = min(3, len(start_planes))
    _rows = math.ceil(len(start_planes) / _columns)
    _width = 4.8 * _columns + 2.4
    _fig, _axes = plt.subplots(_rows, _columns, figsize=(_width, 4.8 * _rows), squeeze=False)
    for _ax, _plane in zip(_axes.flat, start_planes):
        _i, _j = _plane["pair"]
        start_basin_background(_ax, _plane, _i, _j, _n)
        similarity_plane(_ax, _i, _j, [_special], _target_sim, explore_targets, explore_names,
                         velocity_stride.value, _limits)
    for _ax in _axes.flat[len(start_planes):]:
        _ax.axis("off")
    _fig.tight_layout(rect=(0, 0, 1 - 2.4 / _width, 1), w_pad=4, h_pad=3)
    _fig.legend(handles=start_basin_legend(explore_names) + similarity_legend([_special]), loc="center left",
                bbox_to_anchor=(1 - 2.3 / _width, 0.5), frameon=False, fontsize=8)
    _weights = ", ".join(f"{_name} {_w:.2f}" for _name, _w in zip(explore_names, start_weights.value))
    _fig.suptitle(f"start-point basins; specific start point: {_weights}", fontsize=9, y=1.0)
    mo.vstack([_fig, special_figure])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dynamic system

    - As in `spatial_position_scan_dynamic.py`: a run goes on until its visible image (RGBA clipped to 0 … 1) stays
      within *tolerance* (mean squared difference) of its image at step N for *window* steps; then
      `steps_to_stable_image` = N. At least *min steps*, at most *max steps* (then: no stable image)
    - The normal system runs *Length* × `run.t` steps (as `spatial_position_scan_combined.py` runs `STEPS`); its step
      count is `steps_to_pure_attractor`, the step from which the class stays on one target
    - The colour sweeps at the end always run both systems when chosen; the basin map and cube run the dynamic
      system as well only when ticked here (needed for *steps to stable image* in the wheel and cube)
    """)
    return


@app.cell(hide_code=True)
def _():
    dyn_min = mo.ui.number(1, 8192, value=ma.MIN_STEPS, step=1, label="Min steps")
    dyn_max = mo.ui.number(8, 16384, value=ma.MAX_STEPS, step=8, label="Max steps")
    dyn_window = mo.ui.number(1, 2048, value=ma.STABLE_WINDOW, step=1, label="Window")
    dyn_tolerance = mo.ui.dropdown({"1e-3": 1e-3, "3e-4": 3e-4, "1e-4": 1e-4, "3e-5": 3e-5, "1e-5": 1e-5},
                                   value="1e-4", label="Tolerance")
    dyn_fields = mo.ui.checkbox(value=False, label="Also for the basin map and basin cube (slower)")
    mo.hstack([dyn_min, dyn_max, dyn_window, dyn_tolerance, dyn_fields], justify="start", gap=1.5, wrap=True)
    return dyn_fields, dyn_max, dyn_min, dyn_tolerance, dyn_window


@app.cell(hide_code=True)
def _(dyn_max, dyn_min, dyn_tolerance, dyn_window):
    # Settings of the dynamic system, as keyword arguments of ma.rollout_until_stable
    dyn_settings = {"min_steps": int(dyn_min.value), "max_steps": int(dyn_max.value),
                    "window": int(dyn_window.value), "tolerance": float(dyn_tolerance.value)}
    return (dyn_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Basin map

    - One rollout per colour on a polar grid: angle = hue, radius = saturation (or value), the other fixed
    - Left: which target each colour grows into; right: error to the closest target
    - Dots are training seeds (drawn at their own value), the ring is the colour chosen above
    - Coloured lines outline where a stable attractor is reached (criterion above); a hue × saturation map also
      draws them into the wheel
    """)
    return


@app.cell(hide_code=True)
def _():
    basin_slice = mo.ui.radio({"hue × saturation": "s", "hue × value": "v"}, value="hue × saturation",
                              label="Slice", inline=True)
    basin_fixed = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, label="Fixed value / saturation", show_value=True)
    basin_alpha = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, label="Alpha", show_value=True)
    basin_hues = mo.ui.number(4, 180, value=36, step=4, label="Hue steps")
    basin_rings = mo.ui.number(2, 50, value=10, step=1, label="Radial steps")
    basin_run = mo.ui.run_button(label="Compute basin map")
    mo.hstack([basin_slice, basin_fixed, basin_alpha, basin_hues, basin_rings, basin_run],
              justify="start", gap=1.5, wrap=True)
    return (
        basin_alpha,
        basin_fixed,
        basin_hues,
        basin_rings,
        basin_run,
        basin_slice,
    )


@app.cell
def _():
    # Basin maps computed in this session, keyed by their settings
    basin_cache = {}
    return (basin_cache,)


@app.cell(hide_code=True)
def _(
    base_seed,
    basin_alpha,
    basin_cache,
    basin_fixed,
    basin_hues,
    basin_rings,
    basin_run,
    basin_slice,
    dyn_fields,
    dyn_settings,
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_targets,
    get_model,
    memory_budget,
    hidden_base,
    parameter_noise,
):
    _dynamic_key = tuple(sorted(dyn_settings.items())) if dyn_fields.value else None
    _key = (explore_id, parameter_noise.value, basin_slice.value, basin_fixed.value, basin_alpha.value,
            basin_hues.value, basin_rings.value, explore_length.value, hidden_base.value, eval_seed.value, _dynamic_key)
    if basin_run.value and _key not in basin_cache:
        _hue = (np.arange(basin_hues.value) + 0.5) / basin_hues.value
        _radius = (np.arange(basin_rings.value) + 0.5) / basin_rings.value
        _h, _r = np.meshgrid(_hue, _radius, indexing="ij")
        _fixed = np.full_like(_h, basin_fixed.value)
        _s, _v = (_r, _fixed) if basin_slice.value == "s" else (_fixed, _r)
        _hsva = np.stack([_h, _s, _v, np.full_like(_h, basin_alpha.value)], axis=-1).reshape(-1, 4)
        _states = ma.coloured_seeds(base_seed, explore_footprint, ma.hsva_to_rgba(_hsva))
        with mo.status.spinner(f"Rolling out {len(_states)} seed colours"):
            _distances, _rgba, _classes, _changes = ma.rollout(
                get_model(explore_id, parameter_noise.value), _states, explore_targets,
                explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
                memory_gb=memory_budget.value, dynamics=True,
            )
        basin_cache[_key] = {
            "shape": _h.shape, "distances": _distances[:, -1], "final": _rgba[:, -1], "hsva": _hsva,
            "dynamics": ma.attractor_dynamics(_classes, _changes, len(explore_targets)),
            "slice": basin_slice.value, "fixed": basin_fixed.value, "alpha": basin_alpha.value,
        }
        if dyn_fields.value:
            with mo.status.spinner(f"Dynamic system: rolling out {len(_states)} seed colours until stable"):
                _dynamic = ma.rollout_until_stable(
                    get_model(explore_id, parameter_noise.value), _states, explore_targets,
                    jr.PRNGKey(eval_seed.value), memory_gb=memory_budget.value, **dyn_settings,
                )
            basin_cache[_key]["dynamic"] = {_name: _value for _name, _value in _dynamic.items() if _name != "final"}
    # None until computed for these settings (the wheel's boundaries are then cleared)
    basin = basin_cache.get(_key)
    mo.md("Press **Compute basin map** (uses the length, hidden channels, noise and memory settings above)."
          ) if basin is None else None
    return (basin,)


@app.cell(hide_code=True)
def _(basin, show_boundaries, stable_rule):
    # Stable attractor of each basin-map colour (-1 = none) and the boundaries of those regions:
    # basin_lines for the polar maps, wheel_lines for the wheel (only a hue × saturation slice matches it)
    basin_stable = None
    basin_lines, wheel_lines, wheel_note = [], [], ""
    if basin is None:
        if show_boundaries.value:
            wheel_note = "Boundaries: compute the hue × saturation basin map below."
    else:
        basin_stable = ma.stable_attractor(basin["dynamics"], settled=stable_rule.value).reshape(basin["shape"])
        if show_boundaries.value:
            basin_lines = polar_boundary_lines(basin_stable)
            if basin["slice"] == "s":
                wheel_lines = basin_lines
                wheel_note = (f"Boundaries: stable attractors in the basin map at V {basin['fixed']:.2f}, "
                              f"A {basin['alpha']:.2f}.")
            else:
                wheel_note = "Boundaries: need a hue × saturation basin map (this one is hue × value)."
    return basin_lines, basin_stable, wheel_lines, wheel_note


@app.cell(hide_code=True)
def _(
    basin,
    basin_lines,
    basin_slice,
    basin_stable,
    explore_names,
    explore_pair_target,
    explore_threshold_base,
    picker,
    seed_rgba,
    tolerance,
):
    mo.stop(basin is None)
    _threshold = tolerance.value * explore_threshold_base
    _labels = ma.classify(basin["distances"], basin["final"], _threshold)
    _shape = basin["shape"]
    _theta = np.linspace(0, 2 * np.pi, _shape[0] + 1)
    _radius = np.linspace(0, 1, _shape[1] + 1)
    _radial_index = 1 if basin_slice.value == "s" else 2
    _radial_name = "saturation" if basin_slice.value == "s" else "value"

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.6), subplot_kw={"projection": "polar"})
    _axes[0].pcolormesh(_theta, _radius, (_labels.reshape(_shape) + 3).T, cmap=outcome_cmap(len(explore_names)),
                        vmin=-0.5, vmax=len(explore_names) + 2.5, shading="flat")
    _best = np.log10(np.nan_to_num(basin["distances"].min(axis=-1), nan=1e3) + 1e-8).reshape(_shape)
    _mesh = _axes[1].pcolormesh(_theta, _radius, _best.T, cmap="cividis_r", shading="flat")
    _fig.colorbar(_mesh, ax=_axes[1], pad=0.1, shrink=0.8, label="log10 MSE to closest target")
    _axes[1].contour((_theta[:-1] + _theta[1:]) / 2, (_radius[:-1] + _radius[1:]) / 2, _best.T,
                     levels=[np.log10(_threshold)], colors="white", linewidths=1)

    _seed_hsva = ma.rgba_to_hsva(seed_rgba)
    _chosen = ma.rgba_to_hsva([picker.value[_name] for _name in ("r", "g", "b", "a")])
    for _ax in _axes:
        # Stable-attractor boundaries, in the colour of the stable side
        for _h0, _s0, _h1, _s1, _colour, _other in basin_lines:
            _t = np.linspace(_h0, _h1, 8) if _s0 == _s1 else np.array([_h0, _h1])
            _ax.plot(2 * np.pi * _t, np.linspace(_s0, _s1, len(_t)), color="white", lw=3.2, zorder=2,
                     solid_capstyle="round")
            _ax.plot(2 * np.pi * _t, np.linspace(_s0, _s1, len(_t)), color=_colour, lw=1.6, zorder=2,
                     solid_capstyle="round")
        for _pair, _colour in enumerate(_seed_hsva):
            _ax.scatter(2 * np.pi * _colour[0], _colour[_radial_index], s=60, zorder=3,
                        color=outcome_colour(int(explore_pair_target[_pair])), edgecolor="black", lw=1.5)
        _ax.scatter(2 * np.pi * _chosen[0], _chosen[_radial_index], s=140, zorder=4, facecolor="none",
                    edgecolor="black", lw=2)
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0.5, 1.0])
        _ax.set_xticks(np.linspace(0, 2 * np.pi, 6, endpoint=False), [f"hue {_h:.2f}" for _h in np.linspace(0, 1, 6, endpoint=False)],
                       fontsize=7)
    _axes[0].legend(handles=outcome_legend(explore_names), loc="upper left", bbox_to_anchor=(1.05, 1.0),
                    frameon=False, fontsize=8)
    _axes[0].set_title(f"outcome (radius = {_radial_name})", fontsize=9)
    _axes[1].set_title("error to closest target; white = tolerance", fontsize=9)
    _fig.tight_layout()

    # Some final states of each outcome
    _present = [int(_label) for _label in np.unique(_labels)]
    _captions = [f"H{_h:.2f} S{_s:.2f} V{_v:.2f}" for _h, _s, _v, _a in basin["hsva"]]
    _fig2 = outcome_gallery(_labels, basin["final"], _captions, explore_names)

    _counts = {ma.outcome_name(_label, explore_names): int(np.sum(_labels == _label)) for _label in _present}
    _stable_counts = ", ".join(
        f"{explore_names[_k] if _k >= 0 else 'none'} **{int(np.sum(basin_stable == _k))}**"
        for _k in [*range(len(explore_names)), -1]
    )
    mo.vstack([
        _fig,
        mo.md("Outcome counts: " + ", ".join(f"{_name} **{_count}**" for _name, _count in _counts.items())
              + f"<br>Stable attractor (lines): {_stable_counts}"),
        mo.accordion({"Final states by outcome": _fig2}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Basin cube

    - One rollout per colour on a regular grid in the RGB cube, at a fixed alpha
    - Each square is coloured by the target its colour grows into; hover a square for its colour and outcome
    - Dots are training seeds, the ring is the colour chosen above; untick outcomes to look behind them
    - Translucent faces separate cells with a different stable attractor (criterion above), in the stable side's
      colour; where two stable regions meet, the edge has the other one's colour. The same faces appear in the picker
    """)
    return


@app.cell(hide_code=True)
def _():
    cube_steps = mo.ui.number(2, 16, value=8, step=1, label="Steps per axis")
    cube_alpha = mo.ui.slider(0.0, 1.0, value=1.0, step=0.05, label="Alpha", show_value=True)
    cube_run = mo.ui.run_button(label="Compute basin cube")
    mo.hstack([cube_steps, cube_alpha, cube_run], justify="start", gap=1.5, wrap=True)
    return cube_alpha, cube_run, cube_steps


@app.cell
def _():
    # Basin cubes computed in this session, keyed by their settings
    cube_cache = {}
    return (cube_cache,)


@app.cell(hide_code=True)
def _(
    base_seed,
    cube_alpha,
    cube_cache,
    cube_run,
    cube_steps,
    dyn_fields,
    dyn_settings,
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_targets,
    get_model,
    hidden_base,
    memory_budget,
    parameter_noise,
):
    _dynamic_key = tuple(sorted(dyn_settings.items())) if dyn_fields.value else None
    _key = (explore_id, parameter_noise.value, cube_steps.value, cube_alpha.value, explore_length.value,
            hidden_base.value, eval_seed.value, _dynamic_key)
    if cube_run.value and _key not in cube_cache:
        _axis = (np.arange(cube_steps.value) + 0.5) / cube_steps.value
        _r, _g, _b = np.meshgrid(_axis, _axis, _axis, indexing="ij")
        _colours = np.stack([_r, _g, _b, np.full_like(_r, cube_alpha.value)], axis=-1).reshape(-1, 4)
        _colours = _colours.astype(np.float32)
        _states = ma.coloured_seeds(base_seed, explore_footprint, _colours)
        with mo.status.spinner(f"Rolling out {len(_states)} seed colours"):
            _distances, _rgba, _classes, _changes = ma.rollout(
                get_model(explore_id, parameter_noise.value), _states, explore_targets,
                explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
                memory_gb=memory_budget.value, dynamics=True,
            )
        cube_cache[_key] = {
            "distances": _distances[:, -1], "final": _rgba[:, -1], "rgba": _colours, "steps": cube_steps.value,
            "dynamics": ma.attractor_dynamics(_classes, _changes, len(explore_targets)),
        }
        if dyn_fields.value:
            with mo.status.spinner(f"Dynamic system: rolling out {len(_states)} seed colours until stable"):
                _dynamic = ma.rollout_until_stable(
                    get_model(explore_id, parameter_noise.value), _states, explore_targets,
                    jr.PRNGKey(eval_seed.value), memory_gb=memory_budget.value, **dyn_settings,
                )
            cube_cache[_key]["dynamic"] = {_name: _value for _name, _value in _dynamic.items() if _name != "final"}
    # None until computed for these settings (the cube's boundaries are then cleared)
    basin_cube = cube_cache.get(_key)
    mo.md(f"Press **Compute basin cube** ({cube_steps.value ** 3} rollouts; uses the length, hidden channels, "
          "noise and memory settings above).") if basin_cube is None else None
    return (basin_cube,)


@app.cell(hide_code=True)
def _(basin_cube, show_boundaries, stable_rule):
    # Stable attractor of each basin-cube colour (-1 = none) and the faces between those regions
    cube_stable, cube_faces = None, []
    if basin_cube is not None:
        _n = basin_cube["steps"]
        cube_stable = ma.stable_attractor(basin_cube["dynamics"], settled=stable_rule.value).reshape(_n, _n, _n)
        if show_boundaries.value:
            cube_faces = cube_boundary_faces(cube_stable)
    return cube_faces, cube_stable


@app.cell(hide_code=True)
def _(basin, basin_cube, explore_names, field_mode, stable_rule):
    # Wheel and cube coloured by the steps to a stable attractor: cells of the hue × saturation basin map
    # for the wheel, the basin-cube colours as points in the cube, one colour scale for both
    field_cells, field_points, field_scale, field_note = [], [], {}, ""
    if field_mode.value != "colours":
        _system = field_mode.value
        _n = len(explore_names)
        _wheel = (field_steps(basin, _system, stable_rule.value, _n)
                  if basin is not None and basin["slice"] == "s" else None)
        _cube = field_steps(basin_cube, _system, stable_rule.value, _n) if basin_cube is not None else None
        _found = [_values for _values in (_wheel, _cube) if _values is not None]
        if not _found:
            field_note = ("Steps: compute the hue × saturation basin map and/or the basin cube below"
                          + (", with the dynamic system ticked." if _system == "dynamic" else "."))
        else:
            _low, _high = steps_range(*_found)
            if _wheel is not None:
                _hues, _rings = basin["shape"]
                _colours = steps_colours(_wheel, _low, _high)
                field_cells = [[_i / _hues, (_i + 1) / _hues, _j / _rings, (_j + 1) / _rings, _colours[_i * _rings + _j]]
                               for _i in range(_hues) for _j in range(_rings)]
            if _cube is not None:
                _colours = steps_colours(_cube, _low, _high)
                field_points = [
                    [round(float(_r), 3), round(float(_g), 3), round(float(_b), 3), _colour,
                     f"{int(_s)} steps" if np.isfinite(_s) else "no stable attractor"]
                    for (_r, _g, _b, _a), _colour, _s in zip(basin_cube["rgba"], _colours, _cube)
                ]
            _what = "steps to stable attractor" if _system == "normal" else "steps to stable image"
            field_scale = {"title": _what, "low": _low, "high": _high, "colours": list(SEQUENTIAL),
                           "missing": NO_ATTRACTOR_COLOUR, "missing_label": "no stable attractor"}
            _missing = [_part for _part, _values in (("wheel", _wheel), ("cube", _cube)) if _values is None]
            if _missing:
                field_note = (f"Steps in the {' and '.join(_missing)}: compute "
                              + ("the hue × saturation basin map" if "wheel" in _missing else "the basin cube")
                              + (" with the dynamic system ticked." if _system == "dynamic" else "."))
    return field_cells, field_note, field_points, field_scale


@app.cell(hide_code=True)
def _(
    cube_faces,
    field_cells,
    field_note,
    field_points,
    field_scale,
    picker,
    wheel_lines,
    wheel_note,
):
    # Boundaries and steps colouring into the picker. They are set on the widget, so they reach the
    # page without re-creating it, and choosing a colour does not send them back.
    picker.widget.wheel_boundaries = wheel_lines
    picker.widget.wheel_note = " ".join(_note for _note in (wheel_note, field_note) if _note)
    picker.widget.cube_boundaries = cube_faces
    picker.widget.wheel_cells = field_cells
    picker.widget.points = field_points
    picker.widget.scale = field_scale
    return


@app.cell(hide_code=True)
def _(basin_cube, explore_names, explore_threshold_base, tolerance):
    mo.stop(basin_cube is None)
    cube_labels = ma.classify(basin_cube["distances"], basin_cube["final"], tolerance.value * explore_threshold_base)
    _present = [ma.outcome_name(int(_label), explore_names) for _label in np.unique(cube_labels)]
    cube_show = mo.ui.multiselect(_present, value=_present, label="Show outcomes")
    cube_size = mo.ui.slider(2, 16, value=6, step=1, label="Square size", show_value=True)
    mo.hstack([cube_show, cube_size], justify="start", gap=1.5, wrap=True)
    return cube_labels, cube_show, cube_size


@app.cell(hide_code=True)
def _(
    basin_cube,
    cube_faces,
    cube_labels,
    cube_show,
    cube_size,
    cube_stable,
    explore_names,
    field_points,
    field_scale,
    picker,
    seed_markers,
):
    _names = [ma.outcome_name(int(_label), explore_names) for _label in cube_labels]
    # Coloured by outcome, or by steps when the wheel and cube show steps
    _points = [
        [round(float(_r), 3), round(float(_g), 3), round(float(_b), 3), outcome_colour(int(_label)), _name]
        for (_r, _g, _b, _a), _label, _name in zip(basin_cube["rgba"], cube_labels, _names)
    ]
    if field_points:
        _points = [[*_steps[:4], f"{_name}, {_steps[4]}"] for _steps, _name in zip(field_points, _names)]
    _points = [_point for _point, _name in zip(_points, _names) if _name in cube_show.value]
    _view = mo.ui.anywidget(RGBCube(
        r=picker.value["r"], g=picker.value["g"], b=picker.value["b"], a=picker.value["a"], markers=seed_markers,
        points=_points, point_size=float(cube_size.value), pickable=False, view_key="basin",
        cube_boundaries=cube_faces, scale=field_scale if field_points else {},
    ))

    _present = [int(_label) for _label in np.unique(cube_labels)]
    _legend = "<br>".join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{outcome_colour(_label)};'
        f'border:1px solid #555;margin-right:6px"></span>{ma.outcome_name(_label, explore_names)} '
        f"**{int(np.sum(cube_labels == _label))}**"
        for _label in _present
    )
    _captions = [f"R{_r:.2f} G{_g:.2f} B{_b:.2f}" for _r, _g, _b, _a in basin_cube["rgba"]]
    _gallery = outcome_gallery(cube_labels, basin_cube["final"], _captions, explore_names)
    _stable_counts = "<br>".join(
        f"{explore_names[_k] if _k >= 0 else 'none'} **{int(np.sum(cube_stable == _k))}**"
        for _k in [*range(len(explore_names)), -1]
    )
    mo.vstack([
        mo.hstack([_view, mo.md(f"Outcome counts (squares):<br>{_legend}<br><br>"
                                f"Stable attractor (faces):<br>{_stable_counts}")],
                  justify="start", align="start", gap=2),
        mo.accordion({"Final states by outcome": _gallery}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Steps vs. colour sweeps

    - The colour sweeps of the scan scripts, for every pair of targets: the seed pixel goes from the first target's
      seed colour to the second's (*seed colours*), or between their complementary colours (*complementary*),
      alpha 0 … 1. The seed pixel of a target is the middle pixel of its training seed (RGBA); hidden channels from
      *Hidden channels* above (*zeros* is what the scans use)
    - y: the step from which the run stays on one target (normal system, `plot_steps_vs_spectrum_en.py`), or from
      which its image is stable (dynamic system, `plot_steps_vs_spectrum_dynamic.py`); runs without one sit in the
      band above the scale. Colour and shape: the final class
    - Alphas: *coarse alphas* evenly spaced, then *refinement rounds* times *new alphas* in every interval whose two
      ends end in a different class, so the runs crowd where the outcome changes. Below each sweep, zooms into
      these ranges (shaded in the sweep)
    - Seeds sit at the training position; with *all 9 positions* also at the corners and edge midpoints, as in
      the scans, classified after moving the result back to the training position
    """)
    return


@app.cell(hide_code=True)
def _():
    sweep_coarse = mo.ui.number(5, 201, value=21, step=1, label="Coarse alphas")
    sweep_rounds = mo.ui.number(0, 8, value=3, step=1, label="Refinement rounds")
    sweep_points = mo.ui.number(1, 20, value=4, step=1, label="New alphas per interval")
    sweep_systems = mo.ui.multiselect(["normal", "dynamic"], value=["normal", "dynamic"], label="Systems")
    sweep_all_positions = mo.ui.checkbox(value=False, label="All 9 positions (9× the runs)")
    sweep_run = mo.ui.run_button(label="Compute colour sweeps")
    mo.hstack([sweep_coarse, sweep_rounds, sweep_points, sweep_systems, sweep_all_positions, sweep_run],
              justify="start", gap=1.5, wrap=True)
    return sweep_all_positions, sweep_coarse, sweep_points, sweep_rounds, sweep_run, sweep_systems


@app.cell
def _():
    # Colour sweeps computed in this session, keyed by their settings
    sweep_cache = {}
    return (sweep_cache,)


@app.cell(hide_code=True)
def _(
    base_seed,
    dyn_settings,
    eval_seed,
    explore_cfg,
    explore_footprint,
    explore_id,
    explore_length,
    explore_names,
    explore_pair_target,
    explore_seeds,
    explore_targets,
    get_model,
    hidden_base,
    memory_budget,
    parameter_noise,
    sweep_all_positions,
    sweep_cache,
    sweep_coarse,
    sweep_points,
    sweep_rounds,
    sweep_run,
    sweep_systems,
):
    _sweeps = ma.colour_sweeps(explore_seeds, explore_footprint, explore_pair_target, explore_names)
    _all = ma.scan_positions(explore_footprint)
    _positions = _all if sweep_all_positions.value else _all[:1]
    _training = _all[0][1:]
    _circular = str(explore_cfg.model.padding).upper() == "CIRCULAR"
    _steps = explore_length.value * explore_cfg.run.t
    _key = (explore_id, parameter_noise.value, explore_length.value, hidden_base.value, eval_seed.value,
            sweep_coarse.value, sweep_rounds.value, sweep_points.value, tuple(sweep_systems.value),
            sweep_all_positions.value, tuple(sorted(dyn_settings.items())))
    # One sweep per (colour sweep, position)
    _combos = [(_s, _p) for _p in range(len(_positions)) for _s in range(len(_sweeps))]
    if sweep_run.value and _key not in sweep_cache and sweep_systems.value:
        _model = get_model(explore_id, parameter_noise.value)

        def _seeds(index, alpha):
            _pixels = np.array([(1 - _a) * _sweeps[_combos[_i][0]]["start"] + _a * _sweeps[_combos[_i][0]]["end"]
                                for _i, _a in zip(index, alpha)], np.float32)
            _where = [_positions[_combos[_i][1]][1:] for _i in index]
            _states, _shifts = ma.pixel_states(base_seed, _pixels, _where, _training)
            return _states, _shifts, _pixels

        def _normal(index, alpha):
            _states, _shifts, _pixels = _seeds(index, alpha)
            _, _, _classes, _changes = ma.rollout(
                _model, _states, explore_targets, _steps, jr.PRNGKey(eval_seed.value), memory_gb=memory_budget.value,
                dynamics=True, shifts=_shifts, circular=_circular, shared_key=True,
            )
            _dynamics = ma.attractor_dynamics(_classes, _changes, len(explore_targets))
            return {**_dynamics, "rgb": _pixels[:, :3]}

        def _dynamic(index, alpha):
            _states, _shifts, _pixels = _seeds(index, alpha)
            _result = ma.rollout_until_stable(
                _model, _states, explore_targets, jr.PRNGKey(eval_seed.value), memory_gb=memory_budget.value,
                shifts=_shifts, circular=_circular, shared_key=True, **dyn_settings,
            )
            return {**{_name: _value for _name, _value in _result.items() if _name != "final"}, "rgb": _pixels[:, :3]}

        _entry = {"sweeps": _sweeps, "positions": [_name for _name, _, _ in _positions], "combos": _combos,
                  "steps": _steps, "half_width": 1.0 / (sweep_coarse.value - 1)}
        for _system in sweep_systems.value:
            with mo.status.spinner(f"{_system} system: colour sweeps ({len(_combos)} sweeps, "
                                   f"{sweep_coarse.value} alphas each, then refined)"):
                _entry[_system] = ma.adaptive_sweeps(_normal if _system == "normal" else _dynamic, len(_combos),
                                                     coarse=sweep_coarse.value, rounds=sweep_rounds.value,
                                                     points=sweep_points.value)
        sweep_cache[_key] = _entry
    sweep_result = sweep_cache.get(_key)
    mo.md(f"Press **Compute colour sweeps** ({len(_combos)} sweeps × {sweep_coarse.value} alphas to start; uses the "
          "length, hidden channels, noise, memory and dynamic-system settings above)."
          ) if sweep_result is None else None
    return (sweep_result,)


@app.cell(hide_code=True)
def _(sweep_result):
    mo.stop(sweep_result is None)
    sweep_position = mo.ui.dropdown(sweep_result["positions"], value="center", label="Position of the detailed plots")
    sweep_zooms = mo.ui.number(0, 8, value=4, step=1, label="Zooms per sweep (at most)")
    mo.hstack([sweep_position, sweep_zooms], justify="start", gap=1.5) if len(sweep_result["positions"]) > 1 \
        else sweep_zooms
    return sweep_position, sweep_zooms


@app.cell(hide_code=True)
def _(explore_names, stable_rule, sweep_position, sweep_result, sweep_zooms):
    _tabs = {}
    _positions = sweep_result["positions"]
    _here = _positions.index(sweep_position.value)
    for _system in ("normal", "dynamic"):
        if _system not in sweep_result:
            continue
        _results = sweep_result[_system]
        _steps = [sweep_steps(_result, _system, stable_rule.value) for _result in _results]
        # Normal: the scale of the run length; dynamic: of the longest stable run (shared by all its plots)
        _upper = sweep_result["steps"] if _system == "normal" else axis_upper(_steps, default=ma.MIN_STEPS)
        _items = []
        for _k, (_s, _p) in enumerate(sweep_result["combos"]):
            if _p == _here:
                _items.append(sweep_figure(_results[_k], _steps[_k], sweep_result["sweeps"][_s], _system,
                                           explore_names, _upper, sweep_result["half_width"], _positions[_p],
                                           max_zooms=sweep_zooms.value))
        if len(_positions) > 1:
            _grids = []
            for _s, _sweep in enumerate(sweep_result["sweeps"]):
                _by_position = {_positions[_p]: _k for _k, (_ss, _p) in enumerate(sweep_result["combos"]) if _ss == _s}
                _grids.append(positions_figure({_name: _results[_k] for _name, _k in _by_position.items()},
                                               {_name: _steps[_k] for _name, _k in _by_position.items()},
                                               _sweep, _system, explore_names, _upper))
            _items.append(mo.accordion({"All 9 positions": mo.vstack(_grids)}))
        _tabs[f"{_system} system"] = mo.vstack(_items)
    mo.ui.tabs(_tabs)
    return


if __name__ == "__main__":
    app.run()
