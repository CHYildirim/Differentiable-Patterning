"""
Plottet die Zeit bis zum Attraktor gegen das Farbspektrum - aus den CSVs, die
spatial_position_scan_multi.py (oder eine der ..._evolution_*.py-Varianten)
geschrieben hat. Es wird NICHTS neu simuliert; das Skript laeuft in Sekunden.

Erzeugt je Modell:

  steps_vs_spectrum/center/<scan>.png
      y-Achse: Schritt, ab dem das Bild fest einem Attraktor zugeordnet bleibt
               (steps_to_pure_attractor, 0 .. STEPS-1)
      x-Achse: der Farbverlauf des Scans, alpha 0 .. 1, darunter als
               Farbstreifen die tatsaechlichen Startfarben
      Punkte:  Farbe und Form = End-Klasse des Laufs. Laeufe ohne reinen
               Attraktor (INTERMEDIATE, leeres Feld) stehen in einem eigenen
               Band oberhalb der Achse, damit sichtbar bleibt, bei welchen
               Startfarben das passiert.

  steps_vs_spectrum/positions_3x3/<scan>.png
      dasselbe fuer alle 9 Startpositionen, raeumlich angeordnet wie auf dem
      Canvas (oben links = top_left usw.). Nur wenn mehrere Positionen in den
      Daten sind.

  overview_center.png
      alle Farbverlaeufe des Modells auf einen Blick (Mitte).

  rgb_steps_to_pure_attractor.png, rgb_steps_to_stable.png
      die RGB-Schrittkarten mit Farbskala, eigener Farbe fuer "nie erreicht"
      und einem Schluessel, wie die 64 x 64 Felder angeordnet sind.

Und einmal im Ausgabeordner:

  legend_rgb_steps_grayscale.png
      Legende fuer die BESTEHENDEN Graustufenkarten
      rgb_spectrum_steps_to_stable_*.png und
      rgb_spectrum_steps_to_pure_attractor_*.png (schwarz = 0, weiss = STEPS).

Start aus dem Repo-Stammordner:

    python plot_steps_vs_spectrum.py
    python plot_steps_vs_spectrum.py inference_outputs/evolution_intermediate_center
    python plot_steps_vs_spectrum.py <ergebnisordner> <plotordner>
"""

from pathlib import Path
import csv
import sys
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")    # nur Dateien schreiben, kein Fenster

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ======================================================================
# EINSTELLUNGEN
# ======================================================================

# Ergebnisordner eines Scan-Laufs (der OUTPUT_ROOT des Scan-Skripts).
RESULTS_ROOT = Path("inference_outputs/spatial_position_scan_multi_attractor")

# Hierhin kommen die Plots: <PLOT_ROOT>/<name des Ergebnisordners>/<modell>/
PLOT_ROOT = Path("inference_outputs/steps_plots")

# Welche Zeit auf der y-Achse steht.
METRIC = "steps_to_pure_attractor"

DPI = 150

# Farbskala der RGB-Karten: "data" = vom kleinsten bis zum groessten
# vorkommenden Wert (auf 8 gerundet) - die Unterschiede sind dann sichtbar,
# weil die meisten Werte eng beieinander liegen. "full" = immer 0 .. STEPS-1,
# damit sind die Karten verschiedener Modelle direkt vergleichbar.
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
# FARBEN UND STIL
# ======================================================================
# Validiert mit dem Palette-Checker (alle Paare, helle Flaeche): die drei
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
# Die Farbe haengt am Attraktor, nicht an seiner Rolle im Farbverlauf: CRAB
# ist in jedem Bild eines Modells dieselbe Farbe. Diese vier bestehen den
# Palette-Check fuer alle Paare (schlechtestes Paar bei Farbfehlsichtigkeit
# Delta E 9.2, bei normalem Sehen 16.3).
ATTRACTOR_STYLES = [
    ("#2a78d6", "o"),     # blau
    ("#eb6834", "s"),     # orange
    ("#1baf7a", "^"),     # tuerkis
    ("#4a3aa7", "v"),     # violett
]
# Ab dem 5. Attraktor (gibt es bei deinen Modellen nicht) neutral.
EXTRA_ATTRACTOR_STYLE = (INK_SECONDARY, "P")

# Laeufe ohne reinen Attraktor: eigenes Band oberhalb der Skala, neutral.
INTERMEDIATE_STYLE = (INK_SECONDARY, "|")
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


def total_steps(rows):
    """Anzahl Inferenzschritte aus den Daten (Spalte total_steps)."""

    for row in rows:
        value = as_float(row.get("total_steps"))
        if value:
            return int(value)

    return 128


def check_columns(rows, path):

    needed = {METRIC, "classification", "alpha", "scan", "position"}
    missing = needed - set(rows[0]) if rows else needed

    if missing:
        print(
            f"  ! {path}: Spalten fehlen ({', '.join(sorted(missing))}). "
            f"Diese Daten stammen vermutlich von einer aelteren Skriptversion "
            f"ohne Attraktor-Messung - uebersprungen."
        )
        return False

    return True


def series_scans(rows):
    """
    Nur die Farbverlaeufe (Zeilen mit alpha), gruppiert:
    scan -> position -> Zeilen nach alpha sortiert.
    Die Scans bleiben in der Reihenfolge, in der das Scan-Skript sie
    geschrieben hat.
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
    """
    Reihenfolge der Attraktoren des Modells: aus attractors.csv, sonst in der
    Reihenfolge, in der sie in den Scans zuerst auftauchen (das Scan-Skript
    bildet die Paare in Config-Reihenfolge).
    """

    path = model_dir / "attractors.csv"

    if path.is_file():
        ordered = sorted(read_csv(path), key=lambda r: int(r["index"]))
        return [r["name"] for r in ordered]

    names = []

    # Erst die Paar-Endpunkte (in Config-Reihenfolge), dann Klassen, die nur
    # als Ergebnis vorkommen.
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


def reached(row):

    return str(row.get("reaches_pure_attractor", "")).lower() == "true" \
        and row.get(METRIC, "") != ""


def scan_title(rows):

    first = rows[0]
    kind = first.get("scan_type", "")
    a, b = first.get("from_attractor", "?"), first.get("to_attractor", "?")

    if kind == "complementary":
        return f"Komplementaerfarben: {a} → {b}"

    return f"Startfarben: {a} → {b}"


def end_labels(rows):

    first = rows[0]
    a, b = first.get("from_attractor", "?"), first.get("to_attractor", "?")

    if first.get("scan_type") == "complementary":
        return f"Komplement {a}", f"Komplement {b}"

    return a, b


# ======================================================================
# EIN FARBVERLAUF
# ======================================================================

def draw_spectrum_panel(ax, strip_ax, rows, steps, order, compact=False,
                        show_ylabel=True, mark_transitions=False):
    """
    Zeichnet einen Farbverlauf: Punkte auf ax, Startfarben als Streifen auf
    strip_ax. compact=True fuer kleine Felder (Raster, Uebersicht).
    Gibt die vorkommenden End-Klassen zurueck.
    """

    alphas = np.array([float(r["alpha"]) for r in rows])
    band = steps + (24 if compact else 16)      # y-Position "kein Attraktor"
    size = 14 if compact else 30

    # Duenne Verbindungslinie zwischen benachbarten erreichten Punkten, damit
    # der Verlauf lesbar ist. Unterbrochen, wo kein Attraktor erreicht wurde.
    ys = np.array([float(r[METRIC]) if reached(r) else np.nan for r in rows])
    ax.plot(alphas, ys, color=AXIS, linewidth=1.0, zorder=1)

    classes = sorted({r["classification"] for r in rows},
                     key=lambda c: class_sort_key(c, order))

    for classification in classes:

        selected = [r for r in rows if r["classification"] == classification]
        colour, marker, hollow = class_style(classification, order)
        x = [float(r["alpha"]) for r in selected]
        y = [float(r[METRIC]) if reached(r) else band for r in selected]

        if marker == "|":
            # Strich-Marker: Breite ueber linewidths, ohne Ring
            ax.scatter(x, y, s=size * 2.2, marker="|", color=colour,
                       linewidths=1.4, zorder=3)
            continue

        ax.scatter(
            x, y,
            s=size,
            marker=marker,
            facecolors="none" if hollow else colour,
            edgecolors=colour if hollow else SURFACE,
            linewidths=1.2 if hollow else 0.8,    # 0.8 = heller Ring um den Punkt
            zorder=3,
        )

    # Achsen
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-6, band + 10)

    ticks = [t for t in (0, 32, 64, 96, 128) if t <= steps]
    ax.set_yticks(ticks + [band])
    ax.set_yticklabels(
        [str(t) for t in ticks] + ["keiner" if compact else "kein\nAttraktor"]
    )

    for t in ticks:
        ax.axhline(t, color=GRID, linewidth=0.8, zorder=0)

    # Trennlinie zwischen Skala und "kein Attraktor"-Band
    ax.axhline((steps + band) / 2, color=AXIS, linewidth=0.8, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", length=0, labelsize=7 if compact else 9)

    if show_ylabel:
        ax.set_ylabel(f"Schritt, ab dem fest zugeordnet\n(von {steps})")

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
                    x, -4, f" → {after['classification']}  (alpha {x:.3f})",
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
        # Endpunkte als eigene Beschriftung, buendig nach innen, damit nichts
        # ueber den Rand laeuft.
        strip_ax.text(0, -1.9, left, ha="left", va="top", fontsize=8,
                      color=INK_SECONDARY, transform=strip_ax.get_xaxis_transform())
        strip_ax.text(1, -1.9, right, ha="right", va="top", fontsize=8,
                      color=INK_SECONDARY, transform=strip_ax.get_xaxis_transform())
        strip_ax.set_xlabel("alpha (Mischung der Startfarbe)", labelpad=16)

    return set(classes)


def legend_handles(classes, order):

    handles = []

    for classification in sorted(classes, key=lambda c: class_sort_key(c, order)):

        colour, marker, hollow = class_style(classification, order)
        label = {
            INTERMEDIATE: "INTERMEDIATE",
            NO_PATTERN: "leeres Feld",
        }.get(classification, classification)

        if marker == "|":
            handles.append(Line2D([], [], linestyle="none", marker="|",
                                  markersize=9, markeredgewidth=1.4,
                                  color=colour, label=label))
            continue

        handles.append(Line2D(
            [], [], linestyle="none", marker=marker, markersize=6.5,
            markerfacecolor="none" if hollow else colour,
            markeredgecolor=colour if hollow else SURFACE,
            markeredgewidth=1.2 if hollow else 0.8,
            label=label,
        ))

    return handles


def plot_single(path, rows, steps, model, position, order):

    fig = plt.figure(figsize=(8.0, 4.9))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.07], hspace=0.08,
                            left=0.12, right=0.97, top=0.76, bottom=0.17)
    ax = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=ax)

    classes = draw_spectrum_panel(ax, strip, rows, steps, order,
                                  mark_transitions=True)

    reached_rows = [r for r in rows if reached(r)]
    share = len(reached_rows) / len(rows)
    median = (np.median([float(r[METRIC]) for r in reached_rows])
              if reached_rows else None)

    fig.text(0.12, 0.945, f"{model} · {scan_title(rows)} · {position}",
             fontsize=11, color=INK, ha="left")
    fig.text(
        0.12, 0.895,
        f"{share:.0%} der {len(rows)} Laeufe erreichen einen Attraktor"
        + (f", Median Schritt {median:.0f}" if median is not None else ""),
        fontsize=9, color=INK_SECONDARY, ha="left",
    )
    fig.legend(handles=legend_handles(classes, order), loc="upper left",
               bbox_to_anchor=(0.115, 0.875), ncol=len(classes), fontsize=8,
               handletextpad=0.3, columnspacing=1.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_positions_grid(path, positions, steps, model, title, order):

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

            classes |= draw_spectrum_panel(ax, strip, rows, steps, order,
                                           compact=True, show_ylabel=False)

            share = sum(reached(r) for r in rows) / len(rows)
            ax.set_title(f"{name}  ·  {share:.0%} erreicht",
                         fontsize=8.5, loc="left", color=INK)

            if j != 0:
                ax.set_yticklabels([])

    fig.text(0.075, 0.955, f"{model} · {title} · alle Startpositionen",
             fontsize=11, color=INK)
    fig.text(0.075, 0.928,
             "Anordnung wie auf dem Canvas. x: alpha 0 → 1, Streifen = "
             "Startfarbe. y: Schritt, ab dem fest einem Attraktor zugeordnet.",
             fontsize=8.5, color=INK_SECONDARY)
    fig.text(0.012, 0.46, f"Schritt, ab dem fest zugeordnet (von {steps})",
             rotation=90, va="center", fontsize=9, color=INK_SECONDARY)
    fig.legend(handles=legend_handles(classes, order), loc="upper left",
               bbox_to_anchor=(0.07, 0.918), ncol=len(classes), fontsize=8,
               handletextpad=0.3, columnspacing=1.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_overview(path, scans, steps, model, order):
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

            classes |= draw_spectrum_panel(ax, strip, rows, steps, order,
                                           compact=True, show_ylabel=False)
            share = sum(reached(r) for r in rows) / len(rows)
            ax.set_title(f"{scan_title(rows)}  ·  {share:.0%} erreicht",
                         fontsize=8.5, loc="left", color=INK)

            if j != 0:
                ax.set_yticklabels([])

    fig.text(0.075, 1 - 0.32 / height,
             f"{model} · alle Farbverlaeufe · Startposition center",
             fontsize=11, color=INK)
    fig.text(0.075, 1 - 0.56 / height,
             f"y: Schritt, ab dem das Bild fest einem Attraktor zugeordnet "
             f"bleibt (von {steps}). x: alpha 0 → 1, Streifen = Startfarbe.",
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


def colour_range(values, steps):

    finite = values[np.isfinite(values)]

    if RGB_COLOUR_RANGE != "data" or finite.size == 0:
        return 0, steps - 1

    low = int(np.floor(finite.min() / 8) * 8)
    high = int(min(np.ceil((finite.max() + 1) / 8) * 8, steps) - 1)

    if high - low < 16:                      # nie enger als 16 Schritte
        high = min(low + 16, steps - 1)
        low = max(high - 16, 0)

    return low, high


def plot_rgb_map(path, rows, steps, model, column, reached_column, title,
                 not_reached_label):

    size, levels, tiles = rgb_layout(rows)
    values = np.full((size, size), np.nan)

    for r in rows:

        x, y = int(r["x"]), int(r["y"])
        value = as_float(r.get(column))
        ok = (
            str(r.get(reached_column, "")).lower() == "true"
            if reached_column else value is not None
        )

        if ok and value is not None:
            values[y, x] = value

    low, high = colour_range(values, steps)

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

    # Blaustufe direkt in die Kachel schreiben (unten links: dort liegen
    # gruenliche Startfarben, die fast immer einen Attraktor erreichen -
    # die Schrift steht also auf der Skalenfarbe, nicht auf "nie erreicht").
    for t_row in range(tiles):
        for t_col in range(tiles):
            corner = values[t_row * levels + levels - 2,
                            t_col * levels + 1]
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

    fig.text(0.04, 0.945, f"{model} · RGB-Spektrum · {title}",
             fontsize=11, color=INK)
    fig.text(0.04, 0.91,
             f"{size * size} Startfarben an der Mitte · "
             f"{n_missing} davon {not_reached_label}",
             fontsize=9, color=INK_SECONDARY)

    # Farbskala
    cax = fig.add_axes([0.735, 0.50, 0.025, 0.34])
    bar = fig.colorbar(image, cax=cax)
    step = 16 if high - low > 48 else 8
    ticks = {low, high} | set(range((low // step + 1) * step, high, step))
    ticks = sorted(t for t in ticks
                   if t in (low, high) or min(t - low, high - t) >= step / 2)
    bar.set_ticks(ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    cax.set_title(f"Schritt (von {steps})", fontsize=8.5,
                  color=INK_SECONDARY, loc="left", pad=8)
    fig.text(0.735, 0.465,
             "hell = frueh, dunkel = spaet"
             + ("" if (low, high) == (0, steps - 1)
                else f"\nSkala {low}..{high} (Datenbereich)"),
             fontsize=7.5, color=INK_MUTED, va="top")

    fig.legend(
        handles=[Patch(facecolor=NOT_REACHED, edgecolor="none",
                       label=not_reached_label)],
        loc="upper left", bbox_to_anchor=(0.728, 0.43), fontsize=8,
        handlelength=1.2,
    )

    # Schluessel: wie die Felder angeordnet sind (neutral, damit er nicht mit
    # der Schrittskala verwechselt wird)
    fig.text(0.735, 0.355, "Anordnung der Felder", fontsize=8.5, color=INK)
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
    key.text(0.5, -0.22, "Rot", ha="center", va="bottom", fontsize=7,
             color=INK_SECONDARY)
    key.annotate("", xy=(-0.12, 0.95), xytext=(-0.12, 0.05),
                 arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.8),
                 annotation_clip=False)
    key.text(-0.22, 0.5, "Gruen", ha="right", va="center", fontsize=7,
             color=INK_SECONDARY, rotation=90)
    key.set_xticks([])
    key.set_yticks([])

    for side in key.spines.values():
        side.set_visible(False)

    fig.text(
        0.735, 0.03,
        f"Jede Kachel = eine Blaustufe B0..B{levels - 1}.\n"
        f"In jeder Kachel: nach rechts Rot 0..{levels - 1},\n"
        f"nach unten Gruen 0..{levels - 1} (Pfeile an Kachel B0).",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_grayscale_legend(path, steps):
    """Legende fuer die bestehenden Graustufen-PNGs der Scan-Skripte."""

    fig = plt.figure(figsize=(6.4, 1.9))
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.16])
    gradient = np.linspace(0, 1, 256)[None, :]
    ax.imshow(gradient, aspect="auto", cmap="gray", vmin=0, vmax=1,
              extent=(0, steps, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 32, 64, 96, steps] if steps >= 96 else [0, steps])
    ax.tick_params(length=2, color=AXIS, labelsize=8)

    for side in ax.spines.values():
        side.set_visible(False)

    fig.text(0.06, 0.82,
             "Legende: rgb_spectrum_steps_to_*.png (Graustufenkarten)",
             fontsize=10, color=INK)
    fig.text(0.06, 0.68,
             f"Helligkeit = Schritt: schwarz = 0, weiss = {steps}",
             fontsize=8.5, color=INK_SECONDARY)
    fig.text(
        0.06, 0.06,
        "steps_to_pure_attractor: weiss heisst auch 'nie einen reinen "
        "Attraktor erreicht'.\nsteps_to_stable: weiss heisst auch 'bis zum "
        "Ende nicht zur Ruhe gekommen'.",
        fontsize=7.5, color=INK_MUTED,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ======================================================================
# HAUPTPROGRAMM
# ======================================================================

def plot_model(model_dir, out_dir):

    model = model_dir.name
    made = 0
    position_csv = model_dir / "all_position_results.csv"
    steps = 128

    if position_csv.is_file():

        rows = read_csv(position_csv)

        if rows and check_columns(rows, position_csv):

            steps = total_steps(rows)
            order = attractor_order(model_dir, rows)
            scans = series_scans(rows)

            for name, positions in scans.items():

                if "center" in positions:
                    plot_single(
                        out_dir / "steps_vs_spectrum" / "center" / f"{name}.png",
                        positions["center"], steps, model, "center", order,
                    )
                    made += 1

                if len(positions) > 1:
                    any_rows = next(iter(positions.values()))
                    plot_positions_grid(
                        out_dir / "steps_vs_spectrum" / "positions_3x3"
                        / f"{name}.png",
                        positions, steps, model, scan_title(any_rows), order,
                    )
                    made += 1

            if plot_overview(out_dir / "overview_center.png", scans, steps,
                             model, order):
                made += 1

    for rgb_csv in sorted(model_dir.glob("rgb_spectrum_scan*/rgb_spectrum_results.csv")):

        rows = read_csv(rgb_csv)

        if not rows or METRIC not in rows[0]:
            print(f"  ! {rgb_csv}: keine Attraktor-Spalten - uebersprungen")
            continue

        steps = total_steps(rows)
        suffix = "" if rgb_csv.parent.name == "rgb_spectrum_scan" else \
            "_" + rgb_csv.parent.name.replace("rgb_spectrum_scan_", "")

        plot_rgb_map(
            out_dir / f"rgb_steps_to_pure_attractor{suffix}.png", rows, steps,
            model, METRIC, "reaches_pure_attractor",
            "Schritt, ab dem fest einem Attraktor zugeordnet",
            "nie einen reinen Attraktor erreicht",
        )
        plot_rgb_map(
            out_dir / f"rgb_steps_to_stable{suffix}.png", rows, steps, model,
            "steps_to_stable", "converged_before_end",
            "Schritt, ab dem sich der Zustand nicht mehr aendert",
            "bis zum Ende nicht zur Ruhe gekommen",
        )
        made += 2

    return made, steps


def main():

    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_ROOT
    plot_root = Path(sys.argv[2]) if len(sys.argv) > 2 else PLOT_ROOT

    if not results_root.is_dir():
        raise SystemExit(f"Ergebnisordner nicht gefunden: {results_root.resolve()}")

    out_root = plot_root / results_root.name
    models = sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and (
            (p / "all_position_results.csv").is_file()
            or any(p.glob("rgb_spectrum_scan*/rgb_spectrum_results.csv"))
        )
    )

    if not models:
        raise SystemExit(f"Keine Modellordner mit Ergebnissen in {results_root}")

    print(f"Ergebnisse: {results_root.resolve()}")
    print(f"Plots:      {out_root.resolve()}\n")

    steps = 128
    total = 0

    for model_dir in models:
        made, steps = plot_model(model_dir, out_root / model_dir.name)
        total += made
        print(f"  {model_dir.name:24s} {made:4d} Bilder")

    plot_grayscale_legend(out_root / "legend_rgb_steps_grayscale.png", steps)
    total += 1

    print(f"\n{total} Bilder geschrieben nach {out_root.resolve()}")


if __name__ == "__main__":
    main()
