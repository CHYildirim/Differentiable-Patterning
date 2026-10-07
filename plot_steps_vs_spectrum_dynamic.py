"""
DYNAMISCHE VARIANTE von plot_steps_vs_spectrum.py, fuer die Ergebnisse von
spatial_position_scan_dynamic.py (jeder Lauf rechnet bis zum stabilen Bild).

Auf der y-Achse steht jetzt steps_to_stable_image: der Schritt, ab dem sich
das sichtbare Bild nicht mehr aendert. Den Wert hat JEDER Lauf, der ein
stabiles Bild erreicht - auch Laeufe, die als INTERMEDIATE oder als leeres
Feld enden. Diese stehen deshalb jetzt mit ihrer Schrittzahl auf der Skala
(INTERMEDIATE als graue Raute) statt in einem Band ohne Wert. Im Band
oberhalb der Skala stehen nur noch Laeufe, die bis MAX_STEPS kein stabiles
Bild erreicht haben. Die Skala richtet sich nach den Daten (die Laeufe sind
verschieden lang), nicht mehr nach festen 128 Schritten.

Es wird NICHTS neu simuliert; das Skript laeuft in Sekunden.

Erzeugt je Modell:

  steps_vs_spectrum/center/<scan>.png
      y-Achse: Schritt, ab dem das Bild stabil ist (steps_to_stable_image)
      x-Achse: der Farbverlauf des Scans, alpha 0 .. 1, darunter als
               Farbstreifen die tatsaechlichen Startfarben
      Punkte:  Farbe und Form = End-Klasse des Laufs

  steps_vs_spectrum/positions_3x3/<scan>.png
      dasselbe fuer alle 9 Startpositionen, raeumlich angeordnet wie auf dem
      Canvas. Nur wenn mehrere Positionen in den Daten sind.

  overview_center.png
      alle Farbverlaeufe des Modells auf einen Blick (Mitte).

  rgb_steps_to_stable_image.png        (alle End-Klassen, auch INTERMEDIATE)
  rgb_steps_to_pure_attractor.png      (wie bisher, nur reine Attraktoren)
  rgb_steps_to_stable.png              (aenderungsbasiertes Mass, wie bisher)
      RGB-Schrittkarten mit Farbskala, eigener Farbe fuer "nie erreicht" und
      einem Schluessel, wie die Felder angeordnet sind.

Und einmal im Ausgabeordner:

  legend_rgb_steps_grayscale.png
      Legende fuer die Graustufenkarten rgb_spectrum_steps_to_*.png des
      Scan-Skripts (schwarz = 0, weiss = laengster Lauf des Scans).

Sprache der Beschriftung: --lang de (Standard) oder --lang en. Die
englischen Plots gehen in einen eigenen Ordner (..._en).

Start aus dem Repo-Stammordner:

    python plot_steps_vs_spectrum_dynamic.py
    python plot_steps_vs_spectrum_dynamic.py inference_outputs/spatial_position_scan_dynamic/intermediate_center
    python plot_steps_vs_spectrum_dynamic.py <ergebnisordner> <plotordner> --lang en

spatial_position_scan_dynamic.py ruft das Skript am Ende selbst auf
(MAKE_PLOTS / PLOT_LANGUAGES).
"""

from pathlib import Path
import argparse
import csv
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")    # nur Dateien schreiben, kein Fenster

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ======================================================================
# EINSTELLUNGEN
# ======================================================================

# Ergebnisordner eines Scan-Laufs (der OUTPUT_ROOT des Scan-Skripts).
RESULTS_ROOT = Path("inference_outputs/spatial_position_scan_dynamic/evolution_all")

# Hierhin kommen die Plots: <PLOT_ROOT>[_en]/<name des Ergebnisordners>/<modell>/
PLOT_ROOT = Path("inference_outputs/steps_plots_dynamic")

# Welche Zeit auf der y-Achse steht, und welche Spalte sagt, ob der Lauf
# dort ueberhaupt ankommt.
METRIC = "steps_to_stable_image"
REACHED_COLUMN = "stable_image_reached"

DPI = 150

# Farbskala der RGB-Karten: "data" = vom kleinsten bis zum groessten
# vorkommenden Wert (auf 8 gerundet) - die Unterschiede sind dann sichtbar,
# weil die meisten Werte eng beieinander liegen. "full" = immer 0 bis zum
# laengsten Lauf, damit sind die Karten verschiedener Modelle eher
# vergleichbar.
RGB_COLOUR_RANGE = "data"

# Positionen in der Anordnung, in der sie auf dem Canvas liegen.
POSITION_GRID = [
    ["top_left", "top_center", "top_right"],
    ["center_left", "center", "center_right"],
    ["bottom_left", "bottom_center", "bottom_right"],
]

INTERMEDIATE = "INTERMEDIATE"
NO_PATTERN = "NO_VISIBLE_PATTERN"


# ======================================================================
# TEXTE
# ======================================================================

TEXTS = {
    "de": {
        "complementary": "Komplementaerfarben: {a} → {b}",
        "seed_colours": "Startfarben: {a} → {b}",
        "complement_of": "Komplement {name}",
        "band": "nicht\nstabil",
        "band_compact": "nie",
        "ylabel": "Schritt, ab dem das\nBild stabil ist",
        "ylabel_grid": "Schritt, ab dem das Bild stabil ist",
        "xlabel": "alpha (Mischung der Startfarbe)",
        "transition": " → {name}  (alpha {x:.3f})",
        "empty_field": "leeres Feld",
        "share_line": "{share:.0%} der {n} Laeufe erreichen ein stabiles Bild",
        "median": ", Median Schritt {median:.0f}",
        "class_median": " · {name}: Median {median:.0f}",
        "grid_title": "{name}  ·  {share:.0%} stabil",
        "grid_header": "{model} · {title} · alle Startpositionen",
        "grid_sub": (
            "Anordnung wie auf dem Canvas. x: alpha 0 → 1, Streifen = "
            "Startfarbe. y: Schritt, ab dem das Bild stabil ist."
        ),
        "overview_header": "{model} · alle Farbverlaeufe · Startposition center",
        "overview_sub": (
            "y: Schritt, ab dem sich das Bild nicht mehr aendert (alle "
            "End-Klassen, auch INTERMEDIATE). x: alpha 0 → 1, Streifen = "
            "Startfarbe."
        ),
        "rgb_header": "{model} · RGB-Spektrum · {title}",
        "rgb_sub": "{n} Startfarben an der Mitte · {missing} davon {label}",
        "rgb_scale": "Schritt (laengster Lauf: {longest})",
        "rgb_light_dark": "hell = frueh, dunkel = spaet",
        "rgb_range": "\nSkala {low}..{high} (Datenbereich)",
        "rgb_layout": "Anordnung der Felder",
        "rgb_red": "Rot",
        "rgb_green": "Gruen",
        "rgb_tiles": (
            "Jede Kachel = eine Blaustufe B0..B{last}.\n"
            "In jeder Kachel: nach rechts Rot 0..{last},\n"
            "nach unten Gruen 0..{last} (Pfeile an Kachel B0)."
        ),
        "map_stable_image": "Schritt, ab dem das Bild stabil ist",
        "map_stable_image_missing": "bis zum Ende kein stabiles Bild",
        "map_pure": "Schritt, ab dem fest einem Attraktor zugeordnet",
        "map_pure_missing": "nie einen reinen Attraktor erreicht",
        "map_stable": "Schritt, ab dem sich der Zustand nicht mehr aendert",
        "map_stable_missing": "bis zum Ende nicht zur Ruhe gekommen",
        "legend_title": "Legende: rgb_spectrum_steps_to_*.png (Graustufenkarten)",
        "legend_sub": (
            "Helligkeit = Schritt: schwarz = 0, weiss = laengster Lauf des "
            "Scans (steps_map_scale in rgb_spectrum_summary.txt)"
        ),
        "legend_note": (
            "steps_to_stable_image / steps_to_pure_attractor: weiss heisst "
            "auch 'nie erreicht'.\nsteps_to_stable: weiss heisst auch 'bis "
            "zum Ende nicht zur Ruhe gekommen'."
        ),
        "legend_max": "Maximum",
        "skip_columns": (
            "  ! {path}: Spalten fehlen ({missing}). Diese Daten stammen "
            "vermutlich nicht von spatial_position_scan_dynamic.py - "
            "uebersprungen."
        ),
        "skip_rgb": "  ! {path}: keine Spalte {column} - uebersprungen",
        "not_found": "Ergebnisordner nicht gefunden: {path}",
        "no_models": "Keine Modellordner mit Ergebnissen in {path}",
        "results": "Ergebnisse: {path}",
        "plots": "Plots:      {path}\n",
        "per_model": "  {model:24s} {made:4d} Bilder",
        "total": "\n{total} Bilder geschrieben nach {path}",
    },
    "en": {
        "complementary": "Complementary colours: {a} → {b}",
        "seed_colours": "Seed colours: {a} → {b}",
        "complement_of": "complement {name}",
        "band": "not\nstable",
        "band_compact": "never",
        "ylabel": "Step from which the\nimage is stable",
        "ylabel_grid": "Step from which the image is stable",
        "xlabel": "alpha (seed colour mix)",
        "transition": " → {name}  (alpha {x:.3f})",
        "empty_field": "empty canvas",
        "share_line": "{share:.0%} of {n} runs reach a stable image",
        "median": ", median step {median:.0f}",
        "class_median": " · {name}: median {median:.0f}",
        "grid_title": "{name}  ·  {share:.0%} stable",
        "grid_header": "{model} · {title} · all seed positions",
        "grid_sub": (
            "Arranged as on the canvas. x: alpha 0 → 1, strip = seed colour. "
            "y: step from which the image is stable."
        ),
        "overview_header": "{model} · all colour sweeps · seed position center",
        "overview_sub": (
            "y: step from which the image stops changing (all final classes, "
            "INTERMEDIATE included). x: alpha 0 → 1, strip = seed colour."
        ),
        "rgb_header": "{model} · RGB spectrum · {title}",
        "rgb_sub": "{n} seed colours at the center · {missing} of them {label}",
        "rgb_scale": "Step (longest run: {longest})",
        "rgb_light_dark": "light = early, dark = late",
        "rgb_range": "\nscale {low}..{high} (data range)",
        "rgb_layout": "Cell layout",
        "rgb_red": "Red",
        "rgb_green": "Green",
        "rgb_tiles": (
            "Each tile = one blue level B0..B{last}.\n"
            "Inside each tile: red 0..{last} to the right,\n"
            "green 0..{last} downwards (arrows at tile B0)."
        ),
        "map_stable_image": "step from which the image is stable",
        "map_stable_image_missing": "no stable image until the end",
        "map_pure": "step from which locked to one attractor",
        "map_pure_missing": "never reached a pure attractor",
        "map_stable": "step from which the state stops changing",
        "map_stable_missing": "did not settle until the end",
        "legend_title": "Legend: rgb_spectrum_steps_to_*.png (greyscale maps)",
        "legend_sub": (
            "Brightness = step: black = 0, white = longest run of the scan "
            "(steps_map_scale in rgb_spectrum_summary.txt)"
        ),
        "legend_note": (
            "steps_to_stable_image / steps_to_pure_attractor: white also "
            "means 'never reached'.\nsteps_to_stable: white also means 'did "
            "not settle until the end'."
        ),
        "legend_max": "maximum",
        "skip_columns": (
            "  ! {path}: missing columns ({missing}). These data probably do "
            "not come from spatial_position_scan_dynamic.py - skipped."
        ),
        "skip_rgb": "  ! {path}: no column {column} - skipped",
        "not_found": "Results folder not found: {path}",
        "no_models": "No model folders with results in {path}",
        "results": "Results: {path}",
        "plots": "Plots:   {path}\n",
        "per_model": "  {model:24s} {made:4d} figures",
        "total": "\n{total} figures written to {path}",
    },
}

LANGUAGE = "de"


def tr(key, **values):
    """Text in der eingestellten Sprache."""

    text = TEXTS[LANGUAGE][key]

    return text.format(**values) if values else text


# ======================================================================
# FARBEN UND STIL
# ======================================================================
# Validiert mit dem Palette-Checker (alle Paare, helle Flaeche): die
# Kategorien-Farben trennen sich auch bei Farbfehlsichtigkeit (schlechtestes
# Paar Delta E 9.2). Zusaetzlich traegt jede Klasse eine eigene Markerform,
# und es gibt immer eine Legende - die Farbe ist nie die einzige Kennung.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Attraktor k des Modells (Reihenfolge wie in der config) -> (Farbe, Marker).
ATTRACTOR_STYLES = [
    ("#2a78d6", "o"),     # blau
    ("#eb6834", "s"),     # orange
    ("#1baf7a", "^"),     # tuerkis
    ("#4a3aa7", "v"),     # violett
]
# Ab dem 5. Attraktor neutral.
EXTRA_ATTRACTOR_STYLE = (INK_SECONDARY, "P")

# INTERMEDIATE steht jetzt auf der Skala: neutrale graue Raute (eigene
# Form, damit es nicht nur an der Farbe haengt).
INTERMEDIATE_STYLE = (INK_SECONDARY, "D")
EMPTY_STYLE = (INK_SECONDARY, "o")        # als offener Ring

# Sequentielle Skala fuer die RGB-Karten: ein Farbton, hell = wenige
# Schritte, dunkel = viele Schritte.
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
NOT_REACHED = "#eb6834"    # "nie erreicht" in den RGB-Karten

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
})


# ======================================================================
# DATEN LESEN
# ======================================================================

def read_csv(path):

    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def as_float(value):

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reached(row, metric=None, reached_column=None):

    metric = metric or METRIC
    reached_column = reached_column or REACHED_COLUMN

    return str(row.get(reached_column, "")).lower() == "true" \
        and row.get(metric, "") != ""


def longest_run(rows):
    """Laengster Lauf in den Daten (Spalte steps_run)."""

    values = [as_float(row.get("steps_run")) for row in rows]
    values = [v for v in values if v]

    return int(max(values)) if values else 128


def nice_step(upper, max_ticks=6):

    for step in (8, 16, 32, 64, 128, 256, 512, 1024):
        if upper / step <= max_ticks:
            return step

    return 2048


def axis_upper(rows):
    """
    Obergrenze der y-Achse fuer ein Modell: der groesste erreichte Wert,
    auf die Tick-Weite aufgerundet. Alle Felder eines Modells teilen sie,
    damit sie vergleichbar bleiben.
    """

    values = [float(r[METRIC]) for r in rows if reached(r)]

    if not values:
        return 128

    top = max(values)
    step = nice_step(max(top, 1))

    return int(max(np.ceil((top + 1) / step) * step, step))


def check_columns(rows, path):

    needed = {METRIC, REACHED_COLUMN, "classification", "alpha", "scan",
              "position"}
    missing = needed - set(rows[0]) if rows else needed

    if missing:
        print(tr("skip_columns", path=path, missing=", ".join(sorted(missing))))
        return False

    return True


def series_scans(rows):
    """
    Nur die Farbverlaeufe (Zeilen mit alpha), gruppiert:
    scan -> position -> Zeilen nach alpha sortiert.
    """

    scans = OrderedDict()

    for row in rows:

        if row.get("alpha", "") == "":
            continue                     # black / transparent: kein Verlauf

        scans.setdefault(row["scan"], defaultdict(list))[row["position"]].append(row)

    for positions in scans.values():
        for position_rows in positions.values():
            position_rows.sort(key=lambda r: float(r["alpha"]))

    return scans


def attractor_order(model_dir, rows):
    """Reihenfolge der Attraktoren: aus attractors.csv, sonst aus den Daten."""

    path = model_dir / "attractors.csv"

    if path.is_file():
        ordered = sorted(read_csv(path), key=lambda r: int(r["index"]))
        return [r["name"] for r in ordered]

    names = []

    for keys in (("from_attractor", "to_attractor"), ("classification",)):
        for row in rows:
            for key in keys:
                name = row.get(key, "")
                if name and name not in (INTERMEDIATE, NO_PATTERN) \
                        and name not in names:
                    names.append(name)

    return names


def class_style(classification, order):
    """(Farbe, Marker, hohl?) fuer eine End-Klasse."""

    if classification == INTERMEDIATE:
        return INTERMEDIATE_STYLE + (False,)
    if classification == NO_PATTERN:
        return EMPTY_STYLE + (True,)
    if classification in order and order.index(classification) < len(ATTRACTOR_STYLES):
        return ATTRACTOR_STYLES[order.index(classification)] + (False,)

    return EXTRA_ATTRACTOR_STYLE + (False,)


def class_sort_key(classification, order):

    if classification in order:
        return (0, order.index(classification))
    if classification == INTERMEDIATE:
        return (1, 0)
    if classification == NO_PATTERN:
        return (2, 0)

    return (3, classification)


def scan_title(rows):

    first = rows[0]
    kind = first.get("scan_type", "")
    a, b = first.get("from_attractor", "?"), first.get("to_attractor", "?")

    if kind == "complementary":
        return tr("complementary", a=a, b=b)

    return tr("seed_colours", a=a, b=b)


def end_labels(rows):

    first = rows[0]
    a, b = first.get("from_attractor", "?"), first.get("to_attractor", "?")

    if first.get("scan_type") == "complementary":
        return tr("complement_of", name=a), tr("complement_of", name=b)

    return a, b


def class_label(classification):

    if classification == NO_PATTERN:
        return tr("empty_field")

    return classification


# ======================================================================
# EIN FARBVERLAUF
# ======================================================================

def draw_spectrum_panel(ax, strip_ax, rows, upper, order, compact=False,
                        show_ylabel=True, mark_transitions=False):
    """
    Zeichnet einen Farbverlauf: Punkte auf ax, Startfarben als Streifen auf
    strip_ax. upper = Obergrenze der Skala. compact=True fuer kleine Felder.
    Gibt die vorkommenden End-Klassen zurueck.
    """

    alphas = np.array([float(r["alpha"]) for r in rows])
    pad = upper * (0.19 if compact else 0.125)
    band = upper + pad                        # y-Position "nicht stabil"
    size = 14 if compact else 30

    # Duenne Verbindungslinie zwischen benachbarten stabilen Punkten, damit
    # der Verlauf lesbar ist. Unterbrochen, wo kein stabiles Bild entstand.
    ys = np.array([float(r[METRIC]) if reached(r) else np.nan for r in rows])
    ax.plot(alphas, ys, color=AXIS, linewidth=1.0, zorder=1)

    classes = sorted({r["classification"] for r in rows},
                     key=lambda c: class_sort_key(c, order))

    for classification in classes:

        selected = [r for r in rows if r["classification"] == classification]
        colour, marker, hollow = class_style(classification, order)
        x = [float(r["alpha"]) for r in selected]
        y = [float(r[METRIC]) if reached(r) else band for r in selected]

        ax.scatter(
            x, y,
            s=size * (0.8 if marker == "D" else 1.0),
            marker=marker,
            facecolors="none" if hollow else colour,
            edgecolors=colour if hollow else SURFACE,
            linewidths=1.2 if hollow else 0.8,    # 0.8 = heller Ring um den Punkt
            zorder=3,
        )

    # Achsen
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-upper * 0.05, band + pad * 0.6)

    step = nice_step(upper)
    ticks = list(range(0, upper + 1, step))
    ax.set_yticks(ticks + [band])
    ax.set_yticklabels(
        [str(t) for t in ticks] + [tr("band_compact") if compact else tr("band")]
    )

    for t in ticks:
        ax.axhline(t, color=GRID, linewidth=0.8, zorder=0)

    # Trennlinie zwischen Skala und "nicht stabil"-Band
    ax.axhline(upper + pad / 2, color=AXIS, linewidth=0.8, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", length=0, labelsize=7 if compact else 9)

    if show_ylabel:
        ax.set_ylabel(tr("ylabel"))

    # Uebergaenge zwischen End-Klassen sparsam beschriften
    if mark_transitions:

        changes = [
            (rows[i - 1], rows[i]) for i in range(1, len(rows))
            if rows[i]["classification"] != rows[i - 1]["classification"]
        ]

        if 0 < len(changes) <= 4:
            for before, after in changes:
                x = (float(before["alpha"]) + float(after["alpha"])) / 2
                ax.axvline(x, ymax=0.86, color=GRID, linewidth=0.8, zorder=0)
                ax.text(
                    x, -upper * 0.035,
                    tr("transition", name=class_label(after["classification"]), x=x),
                    rotation=90, va="bottom", ha="right",
                    fontsize=7, color=INK_MUTED,
                )

    # Farbstreifen: die echten Startfarben entlang alpha
    colours = np.array([
        [as_float(r.get("R")) or 0.0, as_float(r.get("G")) or 0.0,
         as_float(r.get("B")) or 0.0]
        for r in rows
    ])
    strip_ax.imshow(
        np.clip(colours, 0, 1)[None, :, :],
        aspect="auto",
        extent=(alphas[0] - 0.005, alphas[-1] + 0.005, 0, 1),
        interpolation="nearest",
    )
    strip_ax.set_xlim(ax.get_xlim())
    strip_ax.set_yticks([])

    for side in ("top", "right", "left", "bottom"):
        strip_ax.spines[side].set_visible(False)

    strip_ax.tick_params(axis="x", length=2, color=AXIS,
                         labelsize=7 if compact else 8)

    if compact:
        strip_ax.set_xticks([0, 0.5, 1.0])
        strip_ax.set_xticklabels(["0", "0.5", "1"])
    else:
        strip_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        left, right = end_labels(rows)
        strip_ax.text(0, -1.9, left, ha="left", va="top", fontsize=8,
                      color=INK_SECONDARY, transform=strip_ax.get_xaxis_transform())
        strip_ax.text(1, -1.9, right, ha="right", va="top", fontsize=8,
                      color=INK_SECONDARY, transform=strip_ax.get_xaxis_transform())
        strip_ax.set_xlabel(tr("xlabel"), labelpad=16)

    return set(classes)


def legend_handles(classes, order):

    handles = []

    for classification in sorted(classes, key=lambda c: class_sort_key(c, order)):

        colour, marker, hollow = class_style(classification, order)

        handles.append(Line2D(
            [], [], linestyle="none", marker=marker,
            markersize=5.8 if marker == "D" else 6.5,
            markerfacecolor="none" if hollow else colour,
            markeredgecolor=colour if hollow else SURFACE,
            markeredgewidth=1.2 if hollow else 0.8,
            label=class_label(classification),
        ))

    return handles


def share_text(rows):
    """'93% der 101 Laeufe ...' plus Median und Median je Nicht-Attraktor-Klasse."""

    reached_rows = [r for r in rows if reached(r)]
    share = len(reached_rows) / len(rows)
    text = tr("share_line", share=share, n=len(rows))

    if reached_rows:
        text += tr("median", median=np.median([float(r[METRIC]) for r in reached_rows]))

    for name in (INTERMEDIATE, NO_PATTERN):

        values = [float(r[METRIC]) for r in reached_rows
                  if r["classification"] == name]

        if values:
            text += tr("class_median", name=class_label(name),
                       median=np.median(values))

    return text


def plot_single(path, rows, upper, model, position, order):

    fig = plt.figure(figsize=(8.0, 4.9))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.07], hspace=0.08,
                            left=0.12, right=0.97, top=0.76, bottom=0.17)
    ax = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=ax)

    classes = draw_spectrum_panel(ax, strip, rows, upper, order,
                                  mark_transitions=True)

    fig.text(0.12, 0.945, f"{model} · {scan_title(rows)} · {position}",
             fontsize=11, color=INK, ha="left")
    fig.text(0.12, 0.895, share_text(rows),
             fontsize=9, color=INK_SECONDARY, ha="left")
    fig.legend(handles=legend_handles(classes, order), loc="upper left",
               bbox_to_anchor=(0.115, 0.875), ncol=len(classes), fontsize=8,
               handletextpad=0.3, columnspacing=1.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_positions_grid(path, positions, upper, model, title, order):

    fig = plt.figure(figsize=(11.0, 9.0))
    outer = fig.add_gridspec(3, 3, left=0.075, right=0.985, top=0.86,
                             bottom=0.06, hspace=0.34, wspace=0.10)
    classes = set()

    for i, row_names in enumerate(POSITION_GRID):
        for j, name in enumerate(row_names):

            cell = outer[i, j].subgridspec(2, 1, height_ratios=[1, 0.09],
                                           hspace=0.06)
            ax = fig.add_subplot(cell[0])
            strip = fig.add_subplot(cell[1], sharex=ax)

            rows = positions.get(name)

            if not rows:
                ax.set_axis_off()
                strip.set_axis_off()
                continue

            classes |= draw_spectrum_panel(ax, strip, rows, upper, order,
                                           compact=True, show_ylabel=False)

            share = sum(reached(r) for r in rows) / len(rows)
            ax.set_title(tr("grid_title", name=name, share=share),
                         fontsize=8.5, loc="left", color=INK)

            if j != 0:
                ax.set_yticklabels([])

    fig.text(0.075, 0.955, tr("grid_header", model=model, title=title),
             fontsize=11, color=INK)
    fig.text(0.075, 0.928, tr("grid_sub"), fontsize=8.5, color=INK_SECONDARY)
    fig.text(0.012, 0.46, tr("ylabel_grid"),
             rotation=90, va="center", fontsize=9, color=INK_SECONDARY)
    fig.legend(handles=legend_handles(classes, order), loc="upper left",
               bbox_to_anchor=(0.07, 0.918), ncol=len(classes), fontsize=8,
               handletextpad=0.3, columnspacing=1.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_overview(path, scans, upper, model, order):
    """Alle Farbverlaeufe der Mitte: Zeile = Attraktor-Paar, Spalte = Art."""

    pairs = OrderedDict()

    for name, positions in scans.items():

        rows = positions.get("center")

        if not rows:
            continue

        first = rows[0]
        key = (first.get("from_attractor"), first.get("to_attractor"))
        pairs.setdefault(key, {})[first.get("scan_type", name)] = rows

    if not pairs:
        return False

    kinds = ["normal", "complementary"]
    height = 2.25 * len(pairs) + 1.25
    fig = plt.figure(figsize=(11.0, height))
    outer = fig.add_gridspec(
        len(pairs), 2, left=0.075, right=0.985,
        top=1 - 1.15 / height, bottom=0.3 / height,
        hspace=0.55, wspace=0.10,
    )
    classes = set()

    for i, (_, by_kind) in enumerate(pairs.items()):
        for j, kind in enumerate(kinds):

            cell = outer[i, j].subgridspec(2, 1, height_ratios=[1, 0.1],
                                           hspace=0.06)
            ax = fig.add_subplot(cell[0])
            strip = fig.add_subplot(cell[1], sharex=ax)
            rows = by_kind.get(kind)

            if not rows:
                ax.set_axis_off()
                strip.set_axis_off()
                continue

            classes |= draw_spectrum_panel(ax, strip, rows, upper, order,
                                           compact=True, show_ylabel=False)
            share = sum(reached(r) for r in rows) / len(rows)
            ax.set_title(tr("grid_title", name=scan_title(rows), share=share),
                         fontsize=8.5, loc="left", color=INK)

            if j != 0:
                ax.set_yticklabels([])

    fig.text(0.075, 1 - 0.32 / height, tr("overview_header", model=model),
             fontsize=11, color=INK)
    fig.text(0.075, 1 - 0.56 / height, tr("overview_sub"),
             fontsize=8.5, color=INK_SECONDARY)
    fig.legend(handles=legend_handles(classes, order), loc="upper left",
               bbox_to_anchor=(0.07, 1 - 0.66 / height), ncol=len(classes),
               fontsize=8, handletextpad=0.3, columnspacing=1.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

    return True


# ======================================================================
# RGB-KARTEN MIT LEGENDE
# ======================================================================

def rgb_layout(rows):
    """Grid-Groesse, Stufen und Kacheln aus den Spalten der RGB-CSV."""

    size = max(int(r["x"]) for r in rows) + 1
    levels = max(int(r["red_index"]) for r in rows) + 1
    tiles = size // levels

    return size, levels, tiles


def colour_range(values, longest):

    finite = values[np.isfinite(values)]

    if RGB_COLOUR_RANGE != "data" or finite.size == 0:
        return 0, max(longest, 1)

    low = int(np.floor(finite.min() / 8) * 8)
    high = int(np.ceil((finite.max() + 1) / 8) * 8)

    if high - low < 16:                      # nie enger als 16 Schritte
        high = low + 16

    return low, high


def plot_rgb_map(path, rows, model, column, reached_column, title,
                 not_reached_label):

    size, levels, tiles = rgb_layout(rows)
    values = np.full((size, size), np.nan)
    longest = longest_run(rows)

    for r in rows:

        x, y = int(r["x"]), int(r["y"])
        value = as_float(r.get(column))
        ok = (
            str(r.get(reached_column, "")).lower() == "true"
            if reached_column else value is not None
        )

        if ok and value is not None:
            values[y, x] = value

    low, high = colour_range(values, longest)

    cmap = LinearSegmentedColormap.from_list("steps", SEQUENTIAL)
    cmap.set_bad(NOT_REACHED)

    fig = plt.figure(figsize=(9.6, 7.4))
    ax = fig.add_axes([0.04, 0.06, 0.64, 0.80])
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=low,
                      vmax=high, interpolation="nearest")

    # Kachelgrenzen als schmale Luecke in Flaechenfarbe
    for k in range(1, tiles):
        ax.axhline(k * levels - 0.5, color=SURFACE, linewidth=2.5)
        ax.axvline(k * levels - 0.5, color=SURFACE, linewidth=2.5)

    for t_row in range(tiles):
        for t_col in range(tiles):
            corner = values[t_row * levels + levels - 2 if levels > 1 else 0,
                            t_col * levels + (1 if levels > 1 else 0)]
            dark_bg = np.isfinite(corner) and \
                (corner - low) / max(high - low, 1) > 0.45
            ax.text(t_col * levels + 0.4, t_row * levels + levels - 1.2,
                    f"B{t_row * tiles + t_col}", fontsize=7.5,
                    color=SURFACE if dark_bg else INK, ha="left", va="bottom")

    ax.set_xticks([])
    ax.set_yticks([])

    for side in ax.spines.values():
        side.set_visible(False)

    n_missing = int(np.isnan(values).sum())

    fig.text(0.04, 0.945, tr("rgb_header", model=model, title=title),
             fontsize=11, color=INK)
    fig.text(0.04, 0.91,
             tr("rgb_sub", n=size * size, missing=n_missing,
                label=not_reached_label),
             fontsize=9, color=INK_SECONDARY)

    # Farbskala
    cax = fig.add_axes([0.735, 0.50, 0.025, 0.34])
    bar = fig.colorbar(image, cax=cax)
    step = nice_step(high - low, max_ticks=5)
    ticks = {low, high} | set(range((low // step + 1) * step, high, step))
    ticks = sorted(t for t in ticks
                   if t in (low, high) or min(t - low, high - t) >= step / 2)
    bar.set_ticks(ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    cax.set_title(tr("rgb_scale", longest=longest), fontsize=8.5,
                  color=INK_SECONDARY, loc="left", pad=8)
    fig.text(0.735, 0.465,
             tr("rgb_light_dark")
             + ("" if RGB_COLOUR_RANGE != "data"
                else tr("rgb_range", low=low, high=high)),
             fontsize=7.5, color=INK_MUTED, va="top")

    fig.legend(
        handles=[Patch(facecolor=NOT_REACHED, edgecolor="none",
                       label=not_reached_label)],
        loc="upper left", bbox_to_anchor=(0.728, 0.43), fontsize=8,
        handlelength=1.2,
    )

    # Schluessel: wie die Felder angeordnet sind
    fig.text(0.735, 0.355, tr("rgb_layout"), fontsize=8.5, color=INK)
    key = fig.add_axes([0.745, 0.105, 0.19, 0.19])
    key.set_xlim(0, tiles)
    key.set_ylim(tiles, 0)
    key.set_aspect("equal")

    for t_row in range(tiles):
        for t_col in range(tiles):
            key.add_patch(plt.Rectangle(
                (t_col + 0.05, t_row + 0.05), 0.9, 0.9,
                facecolor=GRID, edgecolor="none",
            ))
            key.text(t_col + 0.5, t_row + 0.52, f"B{t_row * tiles + t_col}",
                     ha="center", va="center", fontsize=7, color=INK)

    key.annotate("", xy=(0.95, -0.12), xytext=(0.05, -0.12),
                 arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.8),
                 annotation_clip=False)
    key.text(0.5, -0.22, tr("rgb_red"), ha="center", va="bottom", fontsize=7,
             color=INK_SECONDARY)
    key.annotate("", xy=(-0.12, 0.95), xytext=(-0.12, 0.05),
                 arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.8),
                 annotation_clip=False)
    key.text(-0.22, 0.5, tr("rgb_green"), ha="right", va="center", fontsize=7,
             color=INK_SECONDARY, rotation=90)
    key.set_xticks([])
    key.set_yticks([])

    for side in key.spines.values():
        side.set_visible(False)

    fig.text(0.735, 0.03, tr("rgb_tiles", last=levels - 1),
             fontsize=7.5, color=INK_SECONDARY, va="bottom")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_grayscale_legend(path):
    """Legende fuer die Graustufen-PNGs des Scan-Skripts."""

    fig = plt.figure(figsize=(6.4, 1.9))
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.16])
    gradient = np.linspace(0, 1, 256)[None, :]
    ax.imshow(gradient, aspect="auto", cmap="gray", vmin=0, vmax=1,
              extent=(0, 1, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", tr("legend_max")])
    ax.tick_params(length=2, color=AXIS, labelsize=8)

    for side in ax.spines.values():
        side.set_visible(False)

    fig.text(0.06, 0.82, tr("legend_title"), fontsize=10, color=INK)
    fig.text(0.06, 0.68, tr("legend_sub"), fontsize=8.5, color=INK_SECONDARY)
    fig.text(0.06, 0.06, tr("legend_note"), fontsize=7.5, color=INK_MUTED)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ======================================================================
# HAUPTPROGRAMM
# ======================================================================

RGB_MAPS = [
    # (Dateiname, Spalte, Spalte "erreicht", Titel, Text fuer "nie")
    ("rgb_steps_to_stable_image", "steps_to_stable_image",
     "stable_image_reached", "map_stable_image", "map_stable_image_missing"),
    ("rgb_steps_to_pure_attractor", "steps_to_pure_attractor",
     "reaches_pure_attractor", "map_pure", "map_pure_missing"),
    ("rgb_steps_to_stable", "steps_to_stable",
     "converged_before_end", "map_stable", "map_stable_missing"),
]


def plot_model(model_dir, out_dir):

    model = model_dir.name
    made = 0
    position_csv = model_dir / "all_position_results.csv"

    if position_csv.is_file():

        rows = read_csv(position_csv)

        if rows and check_columns(rows, position_csv):

            upper = axis_upper(rows)
            order = attractor_order(model_dir, rows)
            scans = series_scans(rows)

            for name, positions in scans.items():

                if "center" in positions:
                    plot_single(
                        out_dir / "steps_vs_spectrum" / "center" / f"{name}.png",
                        positions["center"], upper, model, "center", order,
                    )
                    made += 1

                if len(positions) > 1:
                    any_rows = next(iter(positions.values()))
                    plot_positions_grid(
                        out_dir / "steps_vs_spectrum" / "positions_3x3"
                        / f"{name}.png",
                        positions, upper, model, scan_title(any_rows), order,
                    )
                    made += 1

            if plot_overview(out_dir / "overview_center.png", scans, upper,
                             model, order):
                made += 1

    for rgb_csv in sorted(model_dir.glob("rgb_spectrum_scan*/rgb_spectrum_results.csv")):

        rows = read_csv(rgb_csv)

        if not rows:
            continue

        suffix = "" if rgb_csv.parent.name == "rgb_spectrum_scan" else \
            "_" + rgb_csv.parent.name.replace("rgb_spectrum_scan_", "")

        for filename, column, reached_column, title, missing in RGB_MAPS:

            if column not in rows[0]:
                print(tr("skip_rgb", path=rgb_csv, column=column))
                continue

            plot_rgb_map(
                out_dir / f"{filename}{suffix}.png", rows, model, column,
                reached_column, tr(title), tr(missing),
            )
            made += 1

    return made


def plot_root_for(plot_root, language):
    """Englische Plots in einen eigenen Ordner (<plot_root>_en)."""

    plot_root = Path(plot_root)

    if language == "de":
        return plot_root

    return plot_root.with_name(f"{plot_root.name}_{language}")


def plot_results(results_root=None, plot_root=None, language="de"):
    """Alle Modelle eines Ergebnisordners plotten. Rueckgabe: Anzahl Bilder."""

    global LANGUAGE

    if language not in TEXTS:
        raise ValueError(f"Unknown language {language!r}; use {list(TEXTS)}")

    LANGUAGE = language

    results_root = Path(results_root) if results_root else RESULTS_ROOT
    plot_root = plot_root_for(plot_root or PLOT_ROOT, language)

    if not results_root.is_dir():
        raise FileNotFoundError(tr("not_found", path=results_root.resolve()))

    out_root = plot_root / results_root.name
    models = sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and (
            (p / "all_position_results.csv").is_file()
            or any(p.glob("rgb_spectrum_scan*/rgb_spectrum_results.csv"))
        )
    )

    if not models:
        raise FileNotFoundError(tr("no_models", path=results_root))

    print(tr("results", path=results_root.resolve()))
    print(tr("plots", path=out_root.resolve()))

    total = 0

    for model_dir in models:
        made = plot_model(model_dir, out_root / model_dir.name)
        total += made
        print(tr("per_model", model=model_dir.name, made=made))

    plot_grayscale_legend(out_root / "legend_rgb_steps_grayscale.png")
    total += 1

    print(tr("total", total=total, path=out_root.resolve()))

    return total


def main():

    parser = argparse.ArgumentParser(
        description="Steps until a stable image vs. colour spectrum."
    )
    parser.add_argument("results_root", nargs="?", default=None,
                        help=f"results folder (default: {RESULTS_ROOT})")
    parser.add_argument("plot_root", nargs="?", default=None,
                        help=f"plot folder (default: {PLOT_ROOT}[_en])")
    parser.add_argument("--lang", choices=sorted(TEXTS), default="de",
                        help="language of the figure text (default: de)")
    args = parser.parse_args()

    try:
        plot_results(args.results_root, args.plot_root, args.lang)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
