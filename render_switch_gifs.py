#!/usr/bin/env python3
"""
One GIF for every test that GENUINELY switched in the signal threshold scan
==========================================================================

Reads the results of signal_threshold_scan.py (thresholds.csv, rollouts.csv,
settings.json per model), re-runs the switching rollouts exactly (same start
patterns, same random keys) and saves one labelled GIF per rollout.

"Test" = (stimulus type, stimulus onset t, source -> target, position, duration D).
Depending on --level:
  test    (default) one GIF per test that genuinely switches (status
          "switchable" or "sporadic"), at the smallest switching radius
  radius  one GIF per tested radius at which at least one run genuinely switches
  run     one GIF per individual genuinely switching run (very many)

Each GIF shows:
  - header: model, stimulus type + meaning, source -> target, stimulus onset,
            position, radius, duration, switch rate in the scan, repeat shown
  - left:   the run (before stimulus, stimulus applied, the first 8 steps one by
            one, then every 3rd step, see --frame-every);
            red circle = stimulus area while the stimulus is on, grey afterwards;
            frame colour = current label
  - right:  self-grown references of source and target (fully grown)
  - bottom: steps since stimulus, total time, current label, progress
  - last frame: verdict of the re-run (genuinely switched after n steps)

Usage (from the repo folder, like signal_threshold_scan.py):
  python render_switch_gifs.py
  python render_switch_gifs.py --results inference_outputs/signal_threshold_scan_v2
  python render_switch_gifs.py --models 3attr_circular --times 8,32
  python render_switch_gifs.py --level radius
  python render_switch_gifs.py --max 3          # quick test: 3 GIFs per model
  python render_switch_gifs.py --cpu            # without GPU

Output: <results>/switch_gifs/<model>/t008/<stimulus>__<SOURCE>_to_<TARGET>__<pos>__rNN_DNN__repN.gif
        <results>/switch_gifs/<model>/index.csv
        <results>/switch_gifs/index.html   (gallery with filters, open in a browser)
Existing GIFs are skipped (--overwrite renders them again).
"""

import argparse
import csv
import html
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="inference_outputs/signal_threshold_scan_v2",
                    help="result folder of the scan (with <model>/thresholds.csv)")
    ap.add_argument("--out", default=None, help="output folder (default: <results>/switch_gifs)")
    ap.add_argument("--models", default=None, help="only models whose folder name starts with "
                                                   "this (comma list), e.g. 3attr_circular")
    ap.add_argument("--level", choices=["test", "radius", "run"], default="test")
    ap.add_argument("--status", default="switchable,sporadic",
                    help="which tests (status in thresholds.csv): switchable, sporadic")
    ap.add_argument("--times", default=None, help="only these stimulus onsets, e.g. 8,32")
    ap.add_argument("--signals", default=None, help="only these stimulus types, e.g. graft,seed_cell")
    ap.add_argument("--max", type=int, default=None, help="at most this many GIFs per model")
    ap.add_argument("--frame-every", type=int, default=3,
                    help="one frame every n steps (the first 8 steps after the "
                         "stimulus always every step)")
    ap.add_argument("--colors", type=int, default=255, help="colours in the GIF (fewer = smaller files)")
    ap.add_argument("--scale", type=int, default=2, help="magnification of the run")
    ap.add_argument("--ms", type=int, default=60, help="display time per frame in ms")
    ap.add_argument("--tries", type=int, default=3,
                    help="try this many switching repeats if the re-run "
                         "does not switch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


ARGS = parse_args()
if ARGS.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import jax.random as jr                              # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_threshold_scan as S                    # noqa: E402


# ======================================================================
# Labels
# ======================================================================

SIGNAL_TEXT = {
    "seed_rgba":        "colour = target seed pixel (hidden channels kept)",
    "seed_cell":        "fresh target seed cell (colour, hidden channels = 0)",
    "black_rgba":       "black (hidden channels kept)",
    "black_zero":       "black, hidden channels = 0",
    "transparent_rgba": "transparent (hidden channels kept)",
    "transparent_zero": "hole: all channels = 0",
    "graft":            "graft: all channels = target organism of the same age",
    "graft_hidden":     "graft of the hidden channels only (colour kept)",
    "target_mean":      "all channels = mean target cell",
    "push1":            "cells + 1 x (target - source), channel means",
    "push2":            "cells + 2 x (target - source), channel means",
    "noise":            "noise on all channels (neutral)",
}

# status names in thresholds.csv (written by the scan) -> English
STATUS_EN = {"schaltbar": "switchable", "sporadisch": "sporadic",
             "kein_umschalten": "no_switch"}
STATUS_DE = {v: k for k, v in STATUS_EN.items()}

INK = (11, 11, 11)
INK2 = (82, 81, 77)
MUTED = (163, 162, 156)
WHITE = (255, 255, 255)
RED = (226, 62, 40)
GREY = (150, 150, 150)
COL_SOURCE = (42, 120, 214)       # label = source
COL_TARGET = (27, 150, 100)       # label = target
COL_MIX = (235, 140, 0)           # INTERMEDIATE
COL_OTHER = (150, 80, 200)        # third attractor
COL_EMPTY = (120, 120, 120)       # no pattern
BAD = (200, 90, 20)


def _font(bold=False, size=12):
    try:
        from matplotlib import font_manager as fm
        path = fm.findfont(fm.FontProperties(family="DejaVu Sans",
                                             weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


DENSE_STEPS = 8                      # first steps after the stimulus: every frame
COLORS = max(16, min(255, ARGS.colors))
_MEDIANCUT = getattr(getattr(Image, "Quantize", Image), "MEDIANCUT", 0)
_NODITHER = getattr(getattr(Image, "Dither", Image), "NONE", 0)

F_TITLE = _font(True, 13)
F_TEXT = _font(False, 12)
F_SMALL = _font(False, 11)
F_BIG = _font(True, 14)


def label_color(lab_name, src, tgt_eff, names):
    if lab_name == src:
        return COL_SOURCE
    if tgt_eff and lab_name == tgt_eff:
        return COL_TARGET
    if lab_name == S.INTERMEDIATE:
        return COL_MIX
    if lab_name == S.NO_PATTERN:
        return COL_EMPTY
    if lab_name in names:
        return COL_OTHER
    return INK2


def fit_text(draw, text, font, width):
    """shortens text with '...' if it does not fit into 'width'."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


# ======================================================================
# Read results and select runs
# ======================================================================

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_bundle(path_str):
    p = Path(path_str)
    if (p / "model.eqx").is_file():
        return p
    i = path_str.find("models/")
    if i >= 0 and (Path(path_str[i:]) / "model.eqx").is_file():
        return Path(path_str[i:])
    raise FileNotFoundError(f"model bundle not found: {path_str}")


def to_int(v):
    return int(float(v))


def select_jobs(ctx, groups, rolls, level, n_confirm):
    """List of jobs: per job one condition c, candidate repeats that genuinely
    switched in the scan, and scan statistics."""
    jobs, cache = [], {}
    for g in groups:
        src = ctx.attractors[ctx.names.index(g["source"])]
        tgt = None if g["target"] == S.ANY else ctx.attractors[ctx.names.index(g["target"])]
        pos = (g["position"], to_int(g["y"]), to_int(g["x"]))
        radii = sorted({to_int(r) for r in g["radii_tested"].split()})
        per_r = []
        for r in radii:
            c = S.make_cond(ctx, src, tgt, g["signal"], pos, r, to_int(g["duration"]),
                            to_int(g["t_signal"]), cache)
            runs = rolls.get(c["key"], {})
            strict = sorted(rep for rep, row in runs.items() if row["strict_switch"] == "True")
            per_r.append((r, c, runs, strict))
        if not rolls:
            # without rollouts.csv: use the threshold radius, try all repeats
            if g["r_threshold"] == "":
                continue
            r_t = to_int(g["r_threshold"])
            per_r = [(r, c, {}, list(range(n_confirm))) for r, c, _, _ in per_r if r == r_t]
        hits = [x for x in per_r if x[3]]
        if not hits:
            continue
        if level == "test":
            if g["r_threshold"] != "":
                r_t = to_int(g["r_threshold"])
                hits = [x for x in hits if x[0] == r_t] or hits[:1]
            else:
                hits = hits[:1]
        for r, c, runs, strict in hits:
            n = len(runs)
            stat = dict(n_runs=n, n_strict=len(strict) if runs else None,
                        rate=(len(strict) / n if n else None))
            if level == "run":
                for rep in strict:
                    jobs.append(dict(c=c, group=g, cands=[rep], **stat))
            else:
                jobs.append(dict(c=c, group=g, cands=strict, **stat))
    return jobs


def job_name(job, rep=None):
    c = job["c"]
    tgt = c["target"].replace(S.ANY, "ANY")
    base = (f"{c['signal']}__{c['source']}_to_{tgt}__{c['position']}"
            f"__r{c['radius']:02d}_D{c['duration']:02d}")
    return base if rep is None else f"{base}__rep{rep}"


def t_dir(out_dir, c):
    return out_dir / f"t{c['t_signal']:03d}"


# ======================================================================
# Re-run (like run_jobs in the scan, plus frames)
# ======================================================================

def run_batch(ctx, items, grown, stats, signal_step, ref_compare, every):
    """items: list of (job, rep). Returns (row, frames, labels, pre, painted) per run."""
    dc = ctx.dc
    coefs = [S.signal_coefficients(ctx, j["c"], stats) for j, _ in items]
    x0 = np.stack([grown[(j["c"]["source_index"], r, j["c"]["t_signal"])] for j, r in items])
    G = np.zeros_like(x0)
    for i, ((j, r), cf) in enumerate(zip(items, coefs)):
        if cf[4] == "target":
            G[i] = grown[(j["c"]["target_index"], r, j["c"]["t_signal"])]
        elif cf[4] == "onset":
            G[i] = x0[i]
    cs = [j["c"] for j, _ in items]
    P = dict(
        cy=jnp.asarray([c["y"] for c in cs], jnp.int32),
        cx=jnp.asarray([c["x"] for c in cs], jnp.int32),
        radius=jnp.asarray([c["radius"] for c in cs], jnp.float32),
        duration=jnp.asarray([c["duration"] for c in cs], jnp.int32),
        active=jnp.asarray([True] * len(cs)),
        a=jnp.asarray(np.stack([cf[0] for cf in coefs])),
        b=jnp.asarray(np.stack([cf[1] for cf in coefs])),
        g=jnp.asarray(np.stack([cf[2] for cf in coefs])),
        ns=jnp.asarray(np.stack([cf[3] for cf in coefs])),
        G=jnp.asarray(G),
        key=jnp.stack([S.post_key(r) for _, r in items]),
    )
    ends = np.array([c["duration"] + S.FREE_STEPS for c in cs])
    durs = np.array([c["duration"] for c in cs])
    ends_j = jnp.asarray(ends, jnp.int32)
    dur_j = jnp.asarray(durs, jnp.int32)

    # frame "stimulus applied": same formula as in the scan, without a model step
    painted = []
    for i, (c, cf) in enumerate(zip(cs, coefs)):
        a, b, g, ns, _ = cf
        m = S.disc_mask(ctx, c["y"], c["x"], c["radius"])
        xs = a[:, None, None] * x0[i] + b[:, None, None] + g[:, None, None] * G[i]
        if np.any(ns > 0):
            noise = np.asarray(jr.normal(jr.fold_in(P["key"][i], 0 + 1_000_000), x0.shape[1:]))
            xs = xs + ns[:, None, None] * noise
        painted.append(S.to_rgb_uint8(np.where(m[None], xs, x0[i])[:dc]))
    pre = [S.to_rgb_uint8(x0[i][:dc]) for i in range(len(items))]

    x = jnp.asarray(x0)
    fin = x[:, :dc]
    aft = x[:, :dc]
    frames = [[] for _ in items]
    series = []
    for k in range(int(ends.max())):
        x, lab, frac, _, nact = signal_step(ctx.model, x, jnp.int32(k), P)
        fin = S._capture(fin, x, jnp.int32(k), ends_j)
        aft = S._capture(aft, x, jnp.int32(k), dur_j)
        series.append((lab, frac, nact))
        step = k + 1
        want = [i for i in range(len(items)) if step <= ends[i] and
                (step <= DENSE_STEPS or step % every == 0 or step == ends[i] or step == durs[i])]
        if want:
            host = np.asarray(x[:, :dc])
            for i in want:
                frames[i].append((step, S.to_rgb_uint8(host[i])))
    labels = np.stack([np.asarray(s[0]) for s in series], axis=1)
    fracs = np.stack([np.asarray(s[1]) for s in series], axis=1)
    nacts = np.stack([np.asarray(s[2]) for s in series], axis=1)
    rmse, riou = ref_compare(fin)
    rmse, riou = np.asarray(rmse), np.asarray(riou)
    change = np.asarray(jnp.mean((fin - aft) ** 2, axis=(1, 2, 3)))
    out = []
    for i, (j, r) in enumerate(items):
        row = S.analyse_rollout(labels[i], fracs[i], nacts[i], int(ends[i]), j["c"], ctx,
                                rmse[i], riou[i], change[i])
        out.append((row, frames[i], labels[i][:int(ends[i])], pre[i], painted[i]))
    return out


# ======================================================================
# Draw GIF
# ======================================================================

def render_gif(path, ctx, model_tag, job, rep, row, frames, labels, pre, painted,
               ref_rgb, scale, ms):
    c, g = job["c"], job["group"]
    H, W = ctx.H, ctx.W
    names = ctx.names
    src = c["source"]
    reproduced = bool(row.get("strict_switch"))
    switched_to = row.get("switched_to") or ""
    tgt_eff = c["target"] if c["target"] != S.ANY else (switched_to or None)

    m = 10
    main_w, main_h = W * scale, H * scale
    cap_h = 28                                   # caption above each thumbnail
    thumb = max(60, (main_h - 2 * cap_h - 10) // 2)
    width = m + main_w + 14 + thumb + m
    head_h = 5 * 17 + 10
    foot_h = 40
    height = head_h + main_h + foot_h + 6
    x_main, y_main = m, head_h
    x_side = m + main_w + 14
    end = c["duration"] + S.FREE_STEPS

    # ---- static background: header + references ----
    base = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(base)
    tw = width - 2 * m
    tgt_txt = c["target"] if c["target"] != S.ANY else "any other pattern"
    status = STATUS_EN.get(g.get("status", ""), g.get("status", ""))
    if job.get("n_runs"):
        rate_txt = (f"scan: {job['n_strict']}/{job['n_runs']} runs switched "
                    f"({100 * job['rate']:.0f} %), {status}")
    else:
        rate_txt = f"scan: {status}"
    lines = [
        (F_TITLE, INK, f"{model_tag}  ·  {c['signal']}"),
        (F_TEXT, INK2, SIGNAL_TEXT.get(c["signal"], "")),
        (F_TEXT, INK, f"{src} → {tgt_txt}  ·  stimulus onset t={c['t_signal']}"),
        (F_TEXT, INK, f"position {c['position']}  ·  r={c['radius']} ({c['n_pixels']} pixels)"
                      f"  ·  duration D={c['duration']}"),
        (F_SMALL, INK2, f"{rate_txt}  ·  shown: repeat {rep}"),
    ]
    y = 6
    for font, col, text in lines:
        d.text((m, y), fit_text(d, text, font, tw), font=font, fill=col)
        y += 17

    def side(img, y0, cap1, cap2, col):
        d.text((x_side, y0), cap1, font=F_SMALL, fill=INK2)
        d.text((x_side, y0 + 13), fit_text(d, cap2, F_SMALL, thumb), font=F_SMALL, fill=col)
        im = Image.fromarray(img)
        im = im.resize((thumb, thumb), Image.NEAREST if thumb % W == 0 else Image.LANCZOS)
        base.paste(im, (x_side, y0 + cap_h))
        d.rectangle([x_side - 1, y0 + cap_h - 1, x_side + thumb, y0 + cap_h + thumb], outline=col)

    side(ref_rgb[src], y_main, "Source, no stimulus:", f"{src}", COL_SOURCE)
    if tgt_eff in ref_rgb:
        side(ref_rgb[tgt_eff], y_main + cap_h + thumb + 10, "Target (grown):", f"{tgt_eff}",
             COL_TARGET)

    def run_image(rgb, active):
        im = Image.fromarray(rgb).resize((main_w, main_h), Image.NEAREST)
        dd = ImageDraw.Draw(im)
        r_disp = max(c["radius"] + 0.5, 3.0) * scale
        cx, cy = (c["x"] + 0.5) * scale, (c["y"] + 0.5) * scale
        offs = [(oy, ox) for oy in (-main_h, 0, main_h) for ox in (-main_w, 0, main_w)] \
            if ctx.circular else [(0, 0)]
        for oy, ox in offs:      # CIRCULAR: also the wrapped-around parts
            dd.ellipse([cx + ox - r_disp, cy + oy - r_disp, cx + ox + r_disp, cy + oy + r_disp],
                       outline=RED if active else GREY, width=2 if active else 1)
        return im

    def frame(rgb, step, caption, lab_name, active, verdict=None):
        im = base.copy()
        dd = ImageDraw.Draw(im)
        im.paste(run_image(rgb, active), (x_main, y_main))
        col = label_color(lab_name, src, tgt_eff, names) if lab_name else MUTED
        dd.rectangle([x_main - 3, y_main - 3, x_main + main_w + 2, y_main + main_h + 2],
                     outline=col, width=3)
        yf = y_main + main_h + 7
        dd.text((m, yf), caption, font=F_TEXT, fill=INK)
        if lab_name:
            lx = m + dd.textlength(caption + "   ", font=F_TEXT)
            dd.text((lx, yf), "label:", font=F_TEXT, fill=INK2)
            lx += dd.textlength("label: ", font=F_TEXT)
            dd.text((lx, yf - 1), lab_name, font=F_TITLE, fill=col)
        yb = yf + 21
        dd.rectangle([m, yb, m + main_w, yb + 5], fill=(236, 235, 231))
        dd.rectangle([m, yb, m + int(main_w * min(max(step, 0) / end, 1.0)), yb + 5],
                     fill=COL_SOURCE)
        dmark = m + int(main_w * min(c["duration"] / end, 1.0))
        dd.line([dmark, yb - 3, dmark, yb + 8], fill=RED, width=2)
        if verdict:
            l1, l2, vcol = verdict
            by0 = y_main + main_h - 44
            dd.rectangle([x_main, by0, x_main + main_w - 1, y_main + main_h - 1], fill=vcol)
            dd.text((x_main + 8, by0 + 5), fit_text(dd, l1, F_TITLE, main_w - 16),
                    font=F_TITLE, fill=WHITE)
            dd.text((x_main + 8, by0 + 23), fit_text(dd, l2, F_TEXT, main_w - 16),
                    font=F_TEXT, fill=WHITE)
        return im

    imgs, durs = [], []
    imgs.append(frame(pre, 0, f"before stimulus (t={c['t_signal']})", None, True))
    durs.append(900)
    imgs.append(frame(painted, 0, f"stimulus applied (r={c['radius']})", None, True))
    durs.append(900)
    for step, rgb in frames:
        lab = ctx.class_names[int(labels[step - 1])]
        phase = "stimulus" if step <= c["duration"] else "free"
        cap = f"+{step} ({phase}) · t={c['t_signal'] + step}"
        imgs.append(frame(rgb, step, cap, lab, step <= c["duration"]))
        durs.append(ms)
    step, rgb = frames[-1]
    lab = ctx.class_names[int(labels[step - 1])]
    if reproduced:
        verdict = (f"SWITCHED → {switched_to}",
                   f"after {row.get('steps_to_switch')} steps, shape matches (IoU "
                   f"{row.get('ref_iou_switched', 0):.2f})", COL_TARGET)
    else:
        verdict = ("Re-run: did NOT switch", f"end: {lab}", BAD)
    imgs.append(frame(rgb, step, f"end +{step} · t={c['t_signal'] + step}", lab, False, verdict))
    durs.append(2600)

    # shared palette without dithering -> small files, no flicker
    # (with swatches of all label colours, otherwise rare colours get lost)
    sample = [imgs[0], imgs[1], imgs[len(imgs) // 3], imgs[2 * len(imgs) // 3], imgs[-1]]
    sw, sh = imgs[0].size
    swatch = [INK, INK2, MUTED, WHITE, RED, GREY, COL_SOURCE, COL_TARGET, COL_MIX, COL_OTHER,
              COL_EMPTY, BAD, (236, 235, 231)]
    strip = Image.new("RGB", (sw, sh * len(sample) + 40), WHITE)
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * sh))
    sd = ImageDraw.Draw(strip)
    bw = sw // len(swatch)
    for i, col in enumerate(swatch):
        sd.rectangle([i * bw, sh * len(sample), (i + 1) * bw - 1, sh * len(sample) + 39], fill=col)
    pal = strip.quantize(colors=COLORS, method=_MEDIANCUT, dither=_NODITHER)
    q = [im.quantize(palette=pal, dither=_NODITHER) for im in imgs]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.gif")
    q[0].save(tmp, save_all=True, append_images=q[1:], duration=durs, loop=0,
              optimize=False, disposal=1)
    tmp.replace(path)


# ======================================================================
# Index (CSV + HTML gallery)
# ======================================================================

INDEX_FIELDS = ["file", "model", "signal", "t_signal", "source", "target", "position", "y", "x",
                "radius", "n_pixels", "duration", "status", "scan_n_runs", "scan_n_strict",
                "scan_rate", "rep_shown", "reproduced", "switched_to", "steps_to_switch",
                "final_label"]


def write_index_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_gallery(out_root):
    rows = []
    for p in sorted(out_root.glob("*/index.csv")):
        rows += read_csv(p)
    rows.sort(key=lambda r: (r["model"], int(r["t_signal"]), r["signal"], r["source"],
                             r["target"], int(r["radius"]), int(r["duration"])))
    opts = lambda key: sorted({r[key] for r in rows}, key=lambda v: (len(v), v))
    sel = lambda key, label: (f'<label>{label} <select data-k="{key}"><option value="">all</option>'
                              + "".join(f'<option>{html.escape(v)}</option>' for v in opts(key))
                              + "</select></label>")
    cards = []
    for r in rows:
        ok = r["reproduced"] == "True"
        attrs = " ".join(f'data-{k}="{html.escape(r[k])}"'
                         for k in ("model", "t_signal", "signal", "source", "target", "status"))
        cards.append(
            f'<figure {attrs}><img loading="lazy" src="{html.escape(r["file"])}">'
            f'<figcaption><b>{html.escape(r["signal"])}</b> t={r["t_signal"]} · '
            f'{html.escape(r["source"])} → {html.escape(r["target"])}<br>'
            f'{html.escape(r["position"])} · r={r["radius"]} · D={r["duration"]} · '
            f'scan {r["scan_n_strict"]}/{r["scan_n_runs"]}<br>'
            f'<span class="{"ok" if ok else "bad"}">'
            f'{"switched → " + html.escape(r["switched_to"]) + " after " + r["steps_to_switch"] + " steps" if ok else "re-run did not switch"}'
            f'</span> · {html.escape(r["model"])}</figcaption></figure>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switch GIFs</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafaf8;color:#111}}
.bar{{position:sticky;top:0;background:#fafaf8;padding:8px 0;display:flex;flex-wrap:wrap;gap:12px;
border-bottom:1px solid #ddd;z-index:1}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px;margin-top:12px}}
figure{{margin:0;background:#fff;border:1px solid #e4e3df;border-radius:6px;padding:6px}}
img{{max-width:100%;height:auto;display:block;margin:auto}}
figcaption{{font-size:12px;line-height:1.35;margin-top:4px}}
.ok{{color:#1b9664}} .bad{{color:#c85a14}} #n{{font-weight:600}}
</style></head><body>
<h1>Genuinely switching tests ({len(rows)} GIFs)</h1>
<div class="bar">{sel("model", "Model")}{sel("t_signal", "Stimulus onset t")}{sel("signal", "Stimulus")}
{sel("source", "Source")}{sel("target", "Target")}{sel("status", "Status")}<span id="n"></span></div>
<div class="grid">{"".join(cards)}</div>
<script>
const s=[...document.querySelectorAll('select')],f=[...document.querySelectorAll('figure')];
function u(){{let n=0;f.forEach(e=>{{const v=s.every(x=>!x.value||e.dataset[x.dataset.k]===x.value);
e.style.display=v?'':'none';n+=v}});document.getElementById('n').textContent=n+' shown'}}
s.forEach(x=>x.onchange=u);u();
</script></body></html>"""
    (out_root / "index.html").write_text(page, encoding="utf-8")


# ======================================================================
# One model
# ======================================================================

def run_model(mdir, out_root, emoji_dir):
    model_tag = mdir.name
    settings = json.loads((mdir / "settings.json").read_text())
    rs = settings.get("run_settings", {})
    for k in ("GROW_STEPS", "FREE_STEPS", "SEED", "IOU_MIN", "CHANGE_MIN",
              "ACTIVE_ALPHA_THRESHOLD", "MIXED_MIN_FRACTION"):
        if k in rs:
            setattr(S, k, rs[k])
    t_signals = settings.get("T_SIGNALS", S.T_SIGNALS)
    n_confirm = int(settings.get("REP_CONFIRM", S.REP_CONFIRM))

    wanted_status = {STATUS_DE.get(s.strip(), s.strip()) for s in ARGS.status.split(",")}
    groups = [g for g in read_csv(mdir / "thresholds.csv") if g["status"] in wanted_status]
    if ARGS.times:
        ts = {int(t) for t in ARGS.times.split(",")}
        groups = [g for g in groups if to_int(g["t_signal"]) in ts]
    if ARGS.signals:
        sg = {s.strip() for s in ARGS.signals.split(",")}
        groups = [g for g in groups if g["signal"] in sg]
    print(f"\n==================== {model_tag} ====================")
    print(f"  {len(groups)} tests with status {sorted(STATUS_EN.get(s, s) for s in wanted_status)}")
    if not groups:
        return []

    rolls = defaultdict(dict)
    rp = mdir / "rollouts.csv"
    if rp.is_file():
        for r in read_csv(rp):
            rolls[r["key"]][int(r["repeat"])] = r
    else:
        print("  WARNING: no rollouts.csv -> trying all repeats")

    bundle = find_bundle(rs["bundle"])
    info = S.describe_bundle(S.resolve_bundle(bundle))
    ctx = S.prepare(info, emoji_dir)
    out_dir = out_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = select_jobs(ctx, groups, rolls, ARGS.level, n_confirm)
    jobs.sort(key=lambda j: (j["c"]["t_signal"], j["c"]["signal"], j["c"]["source"],
                             j["c"]["target"], j["c"]["position"], j["c"]["duration"],
                             j["c"]["radius"], j["cands"][0]))
    if ARGS.max:
        jobs = jobs[:ARGS.max]

    # skip existing GIFs
    old_index = {}
    if (out_dir / "index.csv").is_file():
        old_index = {r["file"]: r for r in read_csv(out_dir / "index.csv")}
    index_rows, todo = [], []
    for j in jobs:
        d = t_dir(out_dir, j["c"])
        found = sorted(d.glob(job_name(j) + "__rep*.gif")) if ARGS.level != "run" else \
            sorted(d.glob(job_name(j, j["cands"][0]) + "*.gif"))
        rel = [str(p.relative_to(out_root)) for p in found]
        if found and not ARGS.overwrite and rel[0] in old_index:
            index_rows.append(old_index[rel[0]])
        else:
            for p in found:
                p.unlink()
            todo.append(j)
    print(f"  {len(jobs)} GIFs planned, {len(jobs) - len(todo)} already exist, "
          f"{len(todo)} to render (level '{ARGS.level}')")
    if not todo:
        write_index_csv(out_dir / "index.csv", index_rows)
        return index_rows

    signal_step, grow_step, _ = S.make_step_functions(ctx)
    times = sorted(set(int(t) for t in t_signals) | {S.GROW_STEPS})
    grown = {}

    def ensure_grown(pairs):
        """grows only the missing (attractor, repeat) pairs
        (same start patterns as in the scan: growth_key(a, r))."""
        missing = sorted({p for p in pairs if (p[0], p[1], times[0]) not in grown})
        if not missing:
            return
        print(f"  growing {len(missing)} start patterns to t={S.GROW_STEPS} ...", flush=True)
        t0 = time.time()
        start = 0
        while start < len(missing):
            chunk = missing[start:start + S.BATCH_SIZE]
            try:
                x = jnp.asarray(np.stack([S.seed_state(ctx, ctx.attractors[a]) for a, _ in chunk]))
                keys = jnp.stack([S.growth_key(a, r) for a, r in chunk])
                snaps, t = {}, 0
                for target_t in times:
                    while t < target_t:
                        x = grow_step(ctx.model, x, keys, jnp.int32(t))
                        t += 1
                    snaps[target_t] = np.asarray(x)
            except Exception as exc:
                if not S.is_oom(exc) or S.BATCH_SIZE <= S.MIN_BATCH_SIZE:
                    raise
                x = None
                S.free_gpu_memory()
                S.BATCH_SIZE = max(S.MIN_BATCH_SIZE, S.BATCH_SIZE // 2)
                continue
            for tt, host in snaps.items():
                for i, (a, r) in enumerate(chunk):
                    grown[(a, r, tt)] = host[i]
            start += len(chunk)
        print(f"    done in {time.time() - t0:.1f} s", flush=True)

    def pairs_for(items):
        out = set()
        for j, r in items:
            c = j["c"]
            out.add((c["source_index"], r))
            if c["signal"] in ("graft", "graft_hidden"):
                out.add((c["target_index"], r))
        return out

    ensure_grown({(a.index, 0) for a in ctx.attractors})
    ref = {a.index: grown[(a.index, 0, S.GROW_STEPS)] for a in ctx.attractors}
    ref_compare = S.make_ref_compare(ctx, np.stack([ref[a.index][:ctx.dc] for a in ctx.attractors]))
    ref_rgb = {a.name: S.to_rgb_uint8(ref[a.index][:ctx.dc]) for a in ctx.attractors}
    stats = {(a.index, t): S.body_stats(ctx, grown[(a.index, 0, t)])
             for a in ctx.attractors for t in times}

    # passes: first the first switching repeat of each job; if the re-run
    # does not switch, the next one (up to --tries).
    pending = [(j, 0) for j in todo]
    done, t0 = 0, time.time()
    while pending:
        nxt = []
        start = 0
        while start < len(pending):
            bs = S.BATCH_SIZE
            chunk = pending[start:start + bs]
            items = [(j, j["cands"][a]) for j, a in chunk]
            if len(items) < bs and len(pending) > bs:
                # same batch size -> no recompilation
                items = items + [items[-1]] * (bs - len(items))
            try:
                ensure_grown(pairs_for(items))
                results = run_batch(ctx, items, grown, stats, signal_step, ref_compare,
                                    ARGS.frame_every)[:len(chunk)]
            except Exception as exc:
                if not S.is_oom(exc) or S.BATCH_SIZE <= S.MIN_BATCH_SIZE:
                    raise
                S.free_gpu_memory()
                S.BATCH_SIZE = max(S.MIN_BATCH_SIZE, S.BATCH_SIZE // 2)
                print(f"    GPU memory full -> batch size {S.BATCH_SIZE}", flush=True)
                continue
            for (j, a), (row, frames, labels, pre, painted) in zip(chunk, results):
                rep = j["cands"][a]
                ok = bool(row.get("strict_switch"))
                last = (a + 1 >= min(len(j["cands"]), ARGS.tries))
                if not ok and not last:
                    nxt.append((j, a + 1))
                    continue
                name = job_name(j, rep) + ("" if ok else "__not_reproduced")
                path = t_dir(out_dir, j["c"]) / f"{name}.gif"
                render_gif(path, ctx, model_tag, j, rep, row, frames, labels, pre, painted,
                           ref_rgb, ARGS.scale, ARGS.ms)
                c = j["c"]
                index_rows.append(dict(
                    file=str(path.relative_to(out_root)), model=model_tag, signal=c["signal"],
                    t_signal=c["t_signal"], source=c["source"], target=c["target"],
                    position=c["position"], y=c["y"], x=c["x"], radius=c["radius"],
                    n_pixels=c["n_pixels"], duration=c["duration"],
                    status=STATUS_EN.get(j["group"]["status"], j["group"]["status"]),
                    scan_n_runs=j["n_runs"],
                    scan_n_strict=j["n_strict"], scan_rate=j["rate"], rep_shown=rep,
                    reproduced=ok, switched_to=row.get("switched_to", ""),
                    steps_to_switch=row.get("steps_to_switch", ""),
                    final_label=row.get("final_label", "")))
                done += 1
            start += len(chunk)
            el = time.time() - t0
            print(f"    {done}/{len(todo)} GIFs done ({el / 60:.1f} min)", flush=True)
        pending = nxt
        if pending:
            print(f"  {len(pending)} runs did not switch in the re-run -> "
                  f"next repeat", flush=True)

    index_rows.sort(key=lambda r: r["file"])
    write_index_csv(out_dir / "index.csv", index_rows)
    n_bad = sum(str(r["reproduced"]) != "True" for r in index_rows)
    print(f"  {model_tag}: {len(index_rows)} GIFs in {out_dir}"
          + (f" ({n_bad} not reproduced)" if n_bad else ""))
    grown = None
    S.free_gpu_memory()
    return index_rows


def main():
    results = Path(ARGS.results)
    if not results.is_dir():
        raise SystemExit(f"result folder {results} not found")
    out_root = Path(ARGS.out) if ARGS.out else results / "switch_gifs"
    out_root.mkdir(parents=True, exist_ok=True)
    S.BATCH_SIZE = ARGS.batch_size
    emoji_dir = S.resolve_emoji_dir()
    model_dirs = sorted(p for p in results.iterdir()
                        if (p / "thresholds.csv").is_file() and (p / "settings.json").is_file())
    if ARGS.models:
        wanted = [m.strip() for m in ARGS.models.split(",")]
        model_dirs = [p for p in model_dirs if any(p.name.startswith(w) for w in wanted)]
    print("JAX devices:", jax.devices())
    print("Models:", [p.name for p in model_dirs])
    print("Output:", out_root)
    t_all = time.time()
    for mdir in model_dirs:
        try:
            run_model(mdir, out_root, emoji_dir)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  {mdir.name} skipped because of an error")
        write_gallery(out_root)
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Gallery: {out_root / 'index.html'}")


if __name__ == "__main__":
    main()
