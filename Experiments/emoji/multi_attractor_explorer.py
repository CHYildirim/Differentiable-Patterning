# /// script
# dependencies = [
#   "marimo",
#   "anywidget",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

"""Explore emoji NCAs trained to grow several targets, one per seed.

Run from the repository root with:

    marimo edit Experiments/emoji/multi_attractor_explorer.py

Model bundles are read from the model store (``MODEL_STORE_ROOT``, default
``models/``) and the emoji images from ``DATA_PATH_BASE/Emojis``. Only bundles
with ``data.emoji.task: multi_attractor`` are listed. The evaluation and
rollout helpers are in ``Experiments/emoji/multi_attractor.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
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
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from Experiments.emoji import multi_attractor as ma
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
          element.append(canvas, valueRow, caption("click or drag on the wheel, or click a dot"));
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
            projected.sort((p, q) => p.xyz[2] - q.xyz[2]);
            for (const p of projected) {
              const [x, y] = p.xyz;
              if (p.kind === "point") {
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
          el.append(readout);

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
          for (const name of ["r", "g", "b", "a", "markers", "points", "point_size"]) model.on(`change:${name}`, schedule);
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

    def _choice(column, label):
        _options = sorted(set(runs[column].astype(str)))
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
    for _column, _values in model_filters.value.items():
        _keep &= runs[_column].astype(str).isin(_values).to_numpy()
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
    _controls = [explore_length, explore_repeats] + ([hidden_base] if _has_hidden else [])
    mo.hstack([picker, mo.vstack(_controls)], justify="start", align="start", gap=2)
    return explore_length, explore_repeats, hidden_base, picker, seed_markers


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
    tolerance,
):
    _rgba = np.array([[picker.value[_name] for _name in ("r", "g", "b", "a")]], dtype=np.float32)
    _states = np.repeat(ma.coloured_seeds(base_seed, explore_footprint, _rgba), explore_repeats.value, axis=0)
    _steps = explore_length.value * explore_cfg.run.t
    _n_frames = math.gcd(_steps, 8)
    _distances, _trajectory = ma.rollout(get_model(explore_id, parameter_noise.value), _states, explore_targets,
                                         _steps, jr.PRNGKey(eval_seed.value), frames=_n_frames,
                                         memory_gb=memory_budget.value)
    _threshold = tolerance.value * explore_threshold_base
    _labels = ma.classify(_distances[:, -1], _trajectory[:, -1], _threshold)

    _frames = np.arange(_n_frames + 1) * (_steps // _n_frames)
    _columns = max(len(_frames), explore_repeats.value)
    _fig = plt.figure(figsize=(1.5 * _columns, 6.4))
    _grid = _fig.add_gridspec(3, _columns, height_ratios=[1, 1, 1.6])
    for _i, _frame in enumerate(_frames):
        show_image(_fig.add_subplot(_grid[0, _i]), _trajectory[0, _i], f"step {_frame}")
    for _r in range(explore_repeats.value):
        _label = int(_labels[_r])
        show_image(_fig.add_subplot(_grid[1, _r]), _trajectory[_r, -1],
                   f"rollout {_r}: {ma.outcome_name(_label, explore_names)}", outcome_colour(_label))
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
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Basin map

    - One rollout per colour on a polar grid: angle = hue, radius = saturation (or value), the other fixed
    - Left: which target each colour grows into; right: error to the closest target
    - Dots are training seeds (drawn at their own value), the ring is the colour chosen above
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
    _key = (explore_id, parameter_noise.value, basin_slice.value, basin_fixed.value, basin_alpha.value,
            basin_hues.value, basin_rings.value, explore_length.value, hidden_base.value, eval_seed.value)
    if basin_run.value and _key not in basin_cache:
        _hue = (np.arange(basin_hues.value) + 0.5) / basin_hues.value
        _radius = (np.arange(basin_rings.value) + 0.5) / basin_rings.value
        _h, _r = np.meshgrid(_hue, _radius, indexing="ij")
        _fixed = np.full_like(_h, basin_fixed.value)
        _s, _v = (_r, _fixed) if basin_slice.value == "s" else (_fixed, _r)
        _hsva = np.stack([_h, _s, _v, np.full_like(_h, basin_alpha.value)], axis=-1).reshape(-1, 4)
        _states = ma.coloured_seeds(base_seed, explore_footprint, ma.hsva_to_rgba(_hsva))
        with mo.status.spinner(f"Rolling out {len(_states)} seed colours"):
            _distances, _rgba = ma.rollout(
                get_model(explore_id, parameter_noise.value), _states, explore_targets,
                explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
                memory_gb=memory_budget.value,
            )
        basin_cache[_key] = {"shape": _h.shape, "distances": _distances[:, -1], "final": _rgba[:, -1], "hsva": _hsva}
    mo.stop(_key not in basin_cache, mo.md("Press **Compute basin map** (uses the length, hidden channels, noise and memory settings above)."))
    basin = basin_cache[_key]
    return (basin,)


@app.cell(hide_code=True)
def _(
    basin,
    basin_slice,
    explore_names,
    explore_pair_target,
    explore_threshold_base,
    picker,
    seed_rgba,
    tolerance,
):
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
    mo.vstack([
        _fig,
        mo.md("Outcome counts: " + ", ".join(f"{_name} **{_count}**" for _name, _count in _counts.items())),
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
    _key = (explore_id, parameter_noise.value, cube_steps.value, cube_alpha.value, explore_length.value,
            hidden_base.value, eval_seed.value)
    if cube_run.value and _key not in cube_cache:
        _axis = (np.arange(cube_steps.value) + 0.5) / cube_steps.value
        _r, _g, _b = np.meshgrid(_axis, _axis, _axis, indexing="ij")
        _colours = np.stack([_r, _g, _b, np.full_like(_r, cube_alpha.value)], axis=-1).reshape(-1, 4)
        _colours = _colours.astype(np.float32)
        _states = ma.coloured_seeds(base_seed, explore_footprint, _colours)
        with mo.status.spinner(f"Rolling out {len(_states)} seed colours"):
            _distances, _rgba = ma.rollout(
                get_model(explore_id, parameter_noise.value), _states, explore_targets,
                explore_length.value * explore_cfg.run.t, jr.PRNGKey(eval_seed.value),
                memory_gb=memory_budget.value,
            )
        cube_cache[_key] = {"distances": _distances[:, -1], "final": _rgba[:, -1], "rgba": _colours}
    mo.stop(_key not in cube_cache, mo.md(
        f"Press **Compute basin cube** ({cube_steps.value ** 3} rollouts; uses the length, hidden channels, "
        "noise and memory settings above)."))
    basin_cube = cube_cache[_key]
    return (basin_cube,)


@app.cell(hide_code=True)
def _(basin_cube, explore_names, explore_threshold_base, tolerance):
    cube_labels = ma.classify(basin_cube["distances"], basin_cube["final"], tolerance.value * explore_threshold_base)
    _present = [ma.outcome_name(int(_label), explore_names) for _label in np.unique(cube_labels)]
    cube_show = mo.ui.multiselect(_present, value=_present, label="Show outcomes")
    cube_size = mo.ui.slider(2, 16, value=6, step=1, label="Square size", show_value=True)
    mo.hstack([cube_show, cube_size], justify="start", gap=1.5, wrap=True)
    return cube_labels, cube_show, cube_size


@app.cell(hide_code=True)
def _(basin_cube, cube_labels, cube_show, cube_size, explore_names, picker, seed_markers):
    _names = [ma.outcome_name(int(_label), explore_names) for _label in cube_labels]
    _points = [
        [round(float(_r), 3), round(float(_g), 3), round(float(_b), 3), outcome_colour(int(_label)), _name]
        for (_r, _g, _b, _a), _label, _name in zip(basin_cube["rgba"], cube_labels, _names)
        if _name in cube_show.value
    ]
    _view = mo.ui.anywidget(RGBCube(
        r=picker.value["r"], g=picker.value["g"], b=picker.value["b"], a=picker.value["a"], markers=seed_markers,
        points=_points, point_size=float(cube_size.value), pickable=False, view_key="basin",
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
    mo.vstack([
        mo.hstack([_view, mo.md(f"Outcome counts:<br>{_legend}")], justify="start", align="start", gap=2),
        mo.accordion({"Final states by outcome": _gallery}),
    ])
    return


if __name__ == "__main__":
    app.run()
