"""
EVOLUTIONS-VARIANTE "INTERMEDIATE, NUR MITTE" von spatial_position_scan_multi.py:
Gerechnet wird NUR die zentrale Startposition (SCAN_POSITIONS). Dort: alle
Endbilder (jeder Scan, jedes Alpha, alle 4096 RGB-Farben) und
Entwicklungs-GIFs NUR fuer Laeufe, die als INTERMEDIATE enden.
Die Variante ueber alle 9 Positionen ist
spatial_position_scan_evolution_intermediate.py.

Weil erst nach dem Lauf feststeht, wie er endet, wird in zwei Durchgaengen
gerechnet: zuerst alle Laeufe normal (schnell, ohne Bildaufzeichnung), dann
nur die INTERMEDIATE-Laeufe ein zweites Mal mit Bildaufzeichnung. Die
Laeufe sind deterministisch (gleiche Schluesselfolge), der zweite Durchgang
liefert also denselben Verlauf; das wird je GIF geprueft (Spalte
evolution_gif_replay_matches).

(Weitere Varianten: spatial_position_scan_evolution_center.py,
spatial_position_scan_evolution_all.py.)

Rechnet exakt dieselben Tests (gleiche Schluesselfolge, gleiche Ergebnisse,
gleiche CSVs) und schreibt zusaetzlich:

  * pro Lauf ein ENTWICKLUNGS-GIF vom Seed bis zum Endbild, jeder Schritt
    ein Bild (Seed + STEPS Bilder). Beschriftung je Bild: alpha und
    Position, "step t/STEPS", die Klasse in diesem Schritt, und ab dem
    Schritt, ab dem das Bild einem Attraktor fest zugeordnet bleibt,
    "attractor at N/STEPS".
      <modell>/<position>/<scan>/evolution/alpha_040_microbe.gif
      <modell>/<position>/<scan>/evolution/evolution.gif  (black/transparent)
    Fuer welche Positionen: EVOLUTION_GIF_POSITIONS.
  * RGB-Spektrum-Scan: je Farbe das Endbild als PNG
      <modell>/rgb_spectrum_scan/final_images/r03_g12_b07_crab.png
    und, wenn RGB_SAVE_EVOLUTION_GIFS gesetzt ist, je Farbe ein GIF
      <modell>/rgb_spectrum_scan/evolution/r03_g12_b07_crab.gif
    (r, g, b = Farbstufe 0 .. 15).
  * neue CSV-Spalten evolution_gif bzw. final_image mit dem Pfad relativ
    zum Modellordner, und label_boundary_tie (Lauf lag exakt auf einer
    Entscheidungsgrenze; dann gilt die numpy-Klassifikation).

Vor dem Start wird der Platzbedarf geschaetzt und abgebrochen, wenn auf der
Platte zu wenig frei ist (IGNORE_DISK_SPACE_CHECK).

Das Original spatial_position_scan_multi.py bleibt unveraendert; dieses
Skript schreibt in einen eigenen OUTPUT_ROOT.

----------------------------------------------------------------------

Spatial-Position- und Farb-Scan fuer mehrere Multi-Attraktor-Emoji-NCAs.

Fuer jedes Modell in MODEL_BUNDLES (Abschnitt "HIER DIE PFADE DER 6 MODELLE
EINTRAGEN" weiter unten) laufen dieselben Tests wie im urspruenglichen Skript:

  * 9 Startpositionen (Mitte, Ecken, Kantenmitten)
  * Farbinterpolation zwischen den Startpixeln zweier Attraktoren
    ("normal") und zwischen ihren Komplementaerfarben ("complementary"),
    jetzt fuer jedes Attraktor-Paar
  * schwarzer und transparenter Startpixel
  * RGB-Spektrum-Scan (4096 Farben)

Neu gegenueber dem Original:

  * Attraktoren, Zielbilder und Startpixel werden aus data.emoji.pairs der
    gespeicherten config.yaml gelesen. Der Startpixel wird genau so
    rekonstruiert wie beim Training (Modus full / patch / pixel) und dann
    dessen Mittelpixel genommen.
  * Beliebig viele Attraktoren (2, 3, 4, ...).
  * model.padding wird ausgelesen. Bei CIRCULAR wird fuer die Analyse mit
    Wrap-around ausgerichtet (np.roll), bei REPLICATE ohne (wie bisher).
  * Mehrere Seeds werden gleichzeitig simuliert (jax.vmap + lax.scan).
    Die Zufallsschluessel pro Schritt sind dieselben wie im Original.
  * Alle Modelle laufen nacheinander in einem Aufruf. Faellt ein Modell aus,
    wird der Fehler protokolliert und mit dem naechsten weitergemacht.
  * KONVERGENZ: Fuer JEDEN einzelnen Lauf (jede Position, jedes Alpha, jede
    Spektrumfarbe) wird mitgemessen, wie lange es dauert, bis das stabile
    Endmuster erreicht ist. Dazu wird waehrend des Rollouts nach jedem
    NCA-Schritt festgehalten, wie stark sich der Zustand noch aendert
    (mittlerer quadratischer Unterschied zum vorherigen Schritt). Das Muster
    gilt als stabil ab dem letzten Schritt, nach dem diese Aenderung bis zum
    Ende unter einer Schwelle bleibt. Das kostet keine zusaetzliche
    Rechenzeit, weil es im selben lax.scan mitlaeuft.
    Neue Spalten in allen CSVs (siehe Abschnitt KONVERGENZ in den
    Einstellungen): steps_to_stable, stable_after_fraction,
    converged_before_end, steps_to_stable_abs_tol, residual_change_per_step,
    estimated_seconds_to_stable, ...
    Optional (MEASURE_DISTANCE_TO_FINAL = True, kostet doppelte Rechenzeit)
    wird zusaetzlich exakt gemessen, ab welchem Schritt der Zustand dem
    Endmuster gleicht: steps_to_final_pattern.
  * ATTRAKTOR-KRITERIUM (MEASURE_ATTRACTOR_STEPS, standardmaessig an): der
    Zustand wird nach JEDEM Schritt klassifiziert, nicht nur am Ende.
    steps_to_attractor ist der letzte Schritt, an dem die Klassifikation
    noch von der End-Klassifikation abwich. Das Mass ist robust gegen das
    Zappeln durch fire_rate < 1, weil eine Klassenbezeichnung nicht
    flackert, und liefert auch dann eine brauchbare Zahl, wenn sich der
    Zustand numerisch noch bewegt. Weitere Spalten: attractor_after_fraction,
    attractor_locked_before_end, attractor_label_final,
    attractor_label_switches.
    steps_to_pure_attractor ist die strengere Variante: ab wann ist der
    Zustand einem KONKRETEN Muster zugeordnet und faellt bis zum Ende nicht
    mehr in INTERMEDIATE oder NO_VISIBLE_PATTERN zurueck (und auch nicht auf
    einen anderen Attraktor)? Laeufe, die als Mischung oder als leeres Feld
    enden, bekommen hier keinen Wert, sondern gehen ueber
    reaches_pure_attractor nur in die Quote ein. Der Mittelwert ist deshalb
    immer zusammen mit fraction_reaching_pure_attractor zu lesen - er deckt
    nur die Laeufe ab, die dort ankommen.

WICHTIG zum Zeitfenster: STEPS sollte deutlich groesser sein als das
Training-t des Modells. Wird beides gleich gesetzt (z.B. beides 128), ist
das Muster am Ende des Messfensters gerade erst angekommen und kann dort
kein Plateau bilden - steps_to_stable steht dann bei praktisch jedem Lauf
auf STEPS. steps_to_attractor funktioniert auch in diesem Fall.

Start aus dem Repo-Stammordner (dort, wo Experiments/, Common/ und models/
liegen):

    export DATA_PATH_BASE=/pfad/zu/Data   # Ordner, der "Emojis/" enthaelt
    nohup python spatial_position_scan_multi.py > scan.log 2>&1 &
    tail -f scan.log
"""

from pathlib import Path
import csv
import itertools
import shutil
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from types import SimpleNamespace


# ======================================================================
# CPU / GPU
# ======================================================================
# True  -> alles laeuft auf der CPU, auch wenn eine GPU vorhanden ist.
# False -> JAX entscheidet selbst (GPU, falls vorhanden). Die Wahl kann dann
#          immer noch von aussen erzwungen werden:
#              JAX_PLATFORMS=cpu python spatial_position_scan_multi.py
#
# Dieser Schalter muss VOR "import jax" ausgewertet werden, deshalb steht er
# hier oben und nicht unten bei den uebrigen Einstellungen.
#
# Auf der CPU belegt BATCH_SIZE echten Arbeitsspeicher statt VRAM; bei
# Speicherproblemen BATCH_SIZE weiter unten auf 16 bis 32 verkleinern.
FORCE_CPU = False

if FORCE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"   # aeltere JAX-Versionen
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from Experiments.config_helpers import build_model
from Common.dataloader.emoji import load_emoji_sequence


# ======================================================================
# EINSTELLUNGEN
# ======================================================================

# ######################################################################
# >>>>>>>>>>  HIER DIE PFADE DER 6 MODELLE EINTRAGEN  <<<<<<<<<<
# ######################################################################
#
# Jeder Eintrag ist der BUNDLE-ORDNER eines trainierten Modells, also der
# Ordner, in dem model.eqx und config.yaml liegen. Beispiel (dein altes
# Modell):
#
#   "models/local/emoji_multi_attractor_example_v2/bundles/nca-emoji/"
#   "emoji-multi-attractor-example-v2/20260830T174447Z-cfga63e288e1e9e-db1f45fc"
#
# - Relative Pfade gelten ab dem Repo-Stammordner (dort, wo dieses Skript
#   liegt und gestartet wird). Absolute Pfade funktionieren auch.
# - Zeigt ein Eintrag direkt auf model.eqx oder config.yaml, wird
#   automatisch der Ordner darueber genommen.
# - Die Reihenfolge ist egal. Welches Modell wie viele Attraktoren und welches
#   Padding hat, liest das Skript selbst aus der config.yaml. Die Kommentare
#   hinter den Eintraegen dienen nur deiner Uebersicht.

MODEL_BUNDLES = [
    "models/local/emoji_multi_attractor_example_v2/bundles/nca-emoji/emoji-multi-attractor-example-v2/20260830T174447Z-cfga63e288e1e9e-db1f45fc",   # gNCA, 2 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260909T215800Z-cfg049eedbd4068-d2916c1d",   # gNCA, 2 Attraktoren, CIRCULAR
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260921T212103Z-cfg9b176dd6be39-b584de1c",   # gNCA, 3 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260909T232154Z-cfg23d2f219b302-df559ea2",   # gNCA, 3 Attraktoren, CIRCULAR
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260922T142806Z-cfgee7db3859486-86b9651a",   # gNCA, 4 Attraktoren, REPLICATE
    "models/local/emoji_multi_attractor_2_3_4/bundles/nca-emoji/emoji-multi-attractor-2-3-4/20260910T173700Z-cfgf1ae05f3a10d-be54beee",   # gNCA, 4 Attraktoren, CIRCULAR
]

# ######################################################################
# Ab hier muss normalerweise nichts mehr geaendert werden.
# ######################################################################

PATH_PLACEHOLDER = "HIER_PFAD_EINTRAGEN"

# Nur benutzt, wenn MODEL_BUNDLES eine leere Liste ist: dann werden alle
# Multi-Attraktor-Bundles unter diesem Ordner automatisch gesucht.
MODEL_SEARCH_ROOT = Path("models/local")

# Erwartete Modelle. Weicht der Plan davon ab (falsche Familie, eine
# Kombination fehlt oder ist doppelt), gibt das Skript vor dem Start eine
# deutliche Warnung aus. None bzw. [] schaltet die Pruefung ab.
EXPECTED_FAMILY = "gNCA"
EXPECTED_MODELS = [
    (2, "REPLICATE"), (2, "CIRCULAR"),
    (3, "REPLICATE"), (3, "CIRCULAR"),
    (4, "REPLICATE"), (4, "CIRCULAR"),
]

# ######################################################################
# >>>>>>>>>>  EVOLUTIONS-AUSGABE (neu in diesem Skript)  <<<<<<<<<<
# ######################################################################
#
# Eigener Ordner: dieses Skript ueberschreibt nichts von
# spatial_position_scan_multi.py oder dessen Ergebnissen.
OUTPUT_ROOT = Path("inference_outputs/evolution_intermediate_center")

# Fuer welche Positionen wird pro Lauf ein Entwicklungs-GIF (Seed bis
# Endbild, jeder Schritt ein Bild) geschrieben? Liste von Positionsnamen
# (z.B. ["center"]) oder "all" fuer alle 9 Positionen. Gerechnet und in die
# CSVs geschrieben werden immer alle Positionen - diese Einstellung betrifft
# nur die GIFs.
# Welche Startpositionen ueberhaupt gerechnet werden. "all" = alle 9, sonst
# eine Liste von Namen; "center" muss dabei sein (Referenz fuer die
# Positionsvergleiche). Fuer "alle Endbilder aller Positionen, GIFs nur
# Mitte" hier "all" setzen und EVOLUTION_GIF_POSITIONS = ["center"] lassen.
SCAN_POSITIONS = ["center"]

EVOLUTION_GIF_POSITIONS = ["center"]

# Nur Laeufe mit einer dieser END-Klassen bekommen ein Entwicklungs-GIF.
# Moegliche Werte: "INTERMEDIATE", "NO_VISIBLE_PATTERN" oder ein
# Attraktorname wie "CRAB". None = jeder Lauf bekommt ein GIF.
EVOLUTION_GIF_CLASSES = ["INTERMEDIATE"]

# Nur fuer die Platz-Schaetzung vor dem Start: welcher Anteil der Laeufe
# voraussichtlich ein GIF bekommt. Im letzten vollen Lauf endeten 14.4 %
# aller Laeufe als INTERMEDIATE (an der Mitte: 571 von 4.052
# Positionslaeufen, 3.670 von 24.576 RGB-Farben); 0.25 laesst Luft nach oben.
EVOLUTION_EXPECTED_GIF_FRACTION = 0.25

# RGB-Spektrum-Scan: je Farbe das Endbild als PNG speichern?
RGB_SAVE_FINAL_IMAGES = True
# RGB-Spektrum-Scan: je Farbe zusaetzlich ein Entwicklungs-GIF?
RGB_SAVE_EVOLUTION_GIFS = True

# Jedes wievielte Bild ins GIF (1 = jeder Schritt, also Seed + STEPS Bilder).
EVOLUTION_FRAME_STRIDE = 1
# Anzeigedauer je Bild in ms, und wie lange Seed und Endbild stehen bleiben.
EVOLUTION_FRAME_MS = 60
EVOLUTION_SEED_HOLD_MS = 600
EVOLUTION_END_HOLD_MS = 2000
# Vergroesserung der GIF-Bilder (1 = Originalgroesse des Canvas). Bei 2 ist
# die Beschriftung besser lesbar, die Dateien werden aber groesser.
EVOLUTION_UPSCALE = 1

# Laeufe, deren Zwischenbilder aufgezeichnet werden, laufen in kleineren
# Bloecken: FRAME_BATCH_SIZE x STEPS x 4 x H x W Byte liegen dabei auf der
# GPU (32 x 128 x 4 x 140 x 140 = rund 320 MB).
FRAME_BATCH_SIZE = 32

# Anzahl Threads, die GIFs parallel auf die Platte schreiben.
GIF_WRITER_THREADS = 4

# Schnelle Farbreduktion (FASTOCTREE) beim GIF-Schreiben. Gemessen an einem
# 140 x 140-GIF mit 129 Bildern: rund 8x schneller und halb so gross wie
# Pillows Standardverfahren, mittlere Farbabweichung ~2.6 von 255 statt ~1.5
# - mit blossem Auge nicht zu unterscheiden. False = Pillow-Standard.
EVOLUTION_FAST_QUANTIZE = True

# Vor dem Start wird geschaetzt, wie viel Platz die GIFs brauchen, und
# abgebrochen, wenn auf der Platte deutlich zu wenig frei ist. Die
# Schaetzung ist grob und eher vorsichtig (~8 KB je Bild bei 140 x 140 mit
# FASTOCTREE, abgeleitet aus den bestehenden GIFs). True ueberspringt den
# Abbruch.
IGNORE_DISK_SPACE_CHECK = False
EVOLUTION_BYTES_PER_FRAME_ESTIMATE = 8_000

STEPS = 128
SEED = 0

# Interpolationsfaktor zwischen zwei Startfarben (nicht der Alpha-Kanal).
ALPHAS = np.arange(0, 101) / 100.0

MIXED_MIN_FRACTION = 0.20
ACTIVE_ALPHA_THRESHOLD = 0.10

# Anzahl gleichzeitig simulierter Seeds. Bei Speicherproblemen verkleinern
# (z.B. 32), auf einer grossen GPU vergroessern (z.B. 512).
BATCH_SIZE = 128

# ----------------------------------------------------------------------
# KONVERGENZ: "wie lange bis zum stabilen Endmuster?"
# ----------------------------------------------------------------------
# Nach jedem NCA-Schritt wird gemessen, wie stark sich der Zustand noch
# aendert: change[t] = mean((state[t] - state[t-1])**2).
#
# Das Muster gilt als stabil ab dem letzten Schritt, nach dem change[t] bis
# zum Ende unter der Schwelle bleibt. Die Schwelle ist
#
#     threshold = max(CONVERGENCE_ABS_TOL,
#                     CONVERGENCE_REL_TOL * residual_change)
#
# wobei residual_change der Median der letzten CONVERGENCE_TAIL_STEPS
# Aenderungen ist (das Rauschniveau, das z.B. durch die stochastische
# fire_rate auch im Attraktor bestehen bleibt). Dadurch funktioniert die
# Messung auch bei stochastischen Modellen, bei denen change[t] nie exakt 0
# wird.
CONVERGENCE_ABS_TOL = 1e-5
CONVERGENCE_REL_TOL = 3.0
CONVERGENCE_TAIL_STEPS = 16

# Der Lauf gilt ueberhaupt nur dann als zur Ruhe gekommen, wenn die
# Restaenderung am Ende hoechstens diesen Bruchteil der groessten Aenderung
# des Laufs betraegt. Schwingt das Muster dauerhaft, waere die Schwelle oben
# sonst so gross, dass faelschlich "sofort stabil" herauskaeme. Solche
# Laeufe bekommen steps_to_stable = STEPS und converged_before_end = False.
#
# An echten Modellen gemessen (2attr_replicate, fire_rate 0.5, STEPS = 128)
# liegt residual_over_peak zwischen 0.13 und 0.62. Der frueher benutzte Wert
# 0.01 war an kuenstlichen Testreihen kalibriert und fuer diese Modelle
# unerreichbar streng.
CONVERGENCE_SETTLED_FRACTION = 0.20

# Fensterbreite fuer die Glaettung der Aenderungsreihe (gleitender Median,
# zentriert) vor dem Schwellenvergleich. Bei fire_rate < 1 aktualisiert nur
# ein zufaelliger Teil der Zellen pro Schritt, dadurch zappelt change[t] von
# Schritt zu Schritt. Die Glaettung nimmt dieses Zappeln heraus, ohne den
# Trend zu verschieben. 1 schaltet sie ab.
CONVERGENCE_SMOOTH_WINDOW = 8

# True: Aenderung nur ueber die sichtbaren Kanaele (RGBA) messen.
# False: ueber alle versteckten Kanaele des NCA.
CONVERGENCE_USE_VISIBLE_CHANNELS = True

# Exakte Messung "ab welchem Schritt gleicht der Zustand dem Endmuster?".
# Dafuer muss der Rollout ein zweites Mal laufen (das Endmuster muss vorher
# bekannt sein) -> ungefaehr doppelte Rechenzeit. Standard: aus.
MEASURE_DISTANCE_TO_FINAL = False
CONVERGENCE_DISTANCE_TOL = 1e-4

# ----------------------------------------------------------------------
# ATTRAKTOR-KRITERIUM: "wann liegt das Muster beim End-Attraktor?"
# ----------------------------------------------------------------------
# Zweites, unabhaengiges Mass: der Zustand wird nach JEDEM Schritt so
# klassifiziert wie sonst nur der Endzustand (naechster Attraktor bzw.
# INTERMEDIATE / NO_VISIBLE_PATTERN). steps_to_attractor ist dann der letzte
# Schritt, an dem die Klassifikation noch von der End-Klassifikation abwich.
#
# Vorteil gegenueber der Aenderungs-Messung oben: eine Klassenbezeichnung
# springt vom fire_rate-Rauschen nicht um, das Kriterium liefert also auch
# dann eine brauchbare Zahl, wenn sich der Zustand numerisch noch bewegt.
# Die Klassifikation kostet je Schritt rund ein Tausendstel eines
# NCA-Schritts und laeuft im selben lax.scan mit.
MEASURE_ATTRACTOR_STEPS = True

SAVE_IMAGES = True
SAVE_GIFS = True
RUN_RGB_SPECTRUM_SCAN = True
# Positionen fuer den RGB-Spektrum-Scan (wie bisher nur die Mitte).
SPECTRUM_POSITIONS = ["center"]
# report.txt mit allen Einzelergebnissen (kann bei 4 Attraktoren gross werden).
WRITE_FULL_REPORT = True

# RGB-Spektrum: 64x64 Felder = 16^3 Farben.
SPECTRUM_GRID_SIZE = 64
SPECTRUM_LEVELS = 16
SPECTRUM_TILE_SIZE = 16
SPECTRUM_UPSCALE = 8

INTERMEDIATE = "INTERMEDIATE"
NO_PATTERN = "NO_VISIBLE_PATTERN"

# Farben der Klassifikationskarte. Der n-te Attraktor aus der Config bekommt
# die n-te Farbe; bei 2 Attraktoren ist das wie bisher rot und gruen.
ATTRACTOR_COLOURS = [
    (255, 0, 0, 255),      # 1. Attraktor: rot
    (0, 255, 0, 255),      # 2. Attraktor: gruen
    (255, 0, 255, 255),    # 3. Attraktor: magenta
    (0, 255, 255, 255),    # 4. Attraktor: cyan
    (255, 128, 0, 255),    # 5. Attraktor: orange
    (128, 0, 255, 255),    # 6. Attraktor: violett
]
INTERMEDIATE_COLOUR = (255, 255, 0, 255)   # gelb
NO_PATTERN_COLOUR = (0, 0, 255, 255)       # blau
UNKNOWN_COLOUR = (128, 128, 128, 255)      # grau

ZERO_COMPARISON = {
    "raw_visible_mse_vs_center": 0.0,
    "aligned_visible_mse_vs_center": 0.0,
    "raw_all_channels_mse_vs_center": 0.0,
    "aligned_all_channels_mse_vs_center": 0.0,
}


# ======================================================================
# CONFIG
# ======================================================================

def cfg_value(cfg, *paths, default=None):

    for path in paths:

        value = OmegaConf.select(cfg, path, default=None)

        if value is not None:
            return value

    return default


def plain(value):

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)

    return value


def parse_padding_mode(cfg):

    value = plain(cfg_value(cfg, "model.padding"))

    if isinstance(value, (list, tuple)):

        if len(value) != 1:
            raise ValueError(
                f"model.padding must contain exactly one value, got {value}"
            )

        value = value[0]

    if value is None:
        raise ValueError("model.padding not found in config.yaml")

    return str(value).upper()


def parse_pad(pad):
    """Gibt (oben, unten, links, rechts) zurueck, wie im Original."""

    if pad is None:
        return (0, 0, 0, 0)

    if isinstance(pad, (int, float)):
        p = int(pad)
        return (p, p, p, p)

    pad_list = [int(p) for p in pad]

    if len(pad_list) == 1:
        p = pad_list[0]
        return (p, p, p, p)

    if len(pad_list) == 2:
        return (pad_list[0], pad_list[0], pad_list[1], pad_list[1])

    if len(pad_list) == 4:
        return tuple(pad_list)

    raise ValueError(f"Unexpected padding: {pad_list}")


def read_data_settings(cfg):

    return SimpleNamespace(
        downsample=int(cfg_value(
            cfg,
            "data.preprocessing.downsample",
            "data.downsample",
            "data.emoji.downsample",
            default=1,
        )),
        crop_square=bool(cfg_value(
            cfg,
            "data.augmentation.crop_square",
            "data.emoji.crop_square",
            "data.preprocessing.crop_square",
            default=False,
        )),
        pad=parse_pad(plain(cfg_value(
            cfg,
            "data.augmentation.pad",
            "data.emoji.pad",
            default=0,
        ))),
        data_channels=int(cfg_value(
            cfg,
            "data.augmentation.data_channels",
            "data.emoji.data_channels",
            "data.data_channels",
            default=4,
        )),
        iterations=cfg_value(
            cfg, "run.iterations", "training.loop.iterations", default="?"
        ),
        run_t=cfg_value(cfg, "run.t", "training.loop.t", default="?"),
    )


RESERVED_NAMES = {INTERMEDIATE, NO_PATTERN}


def attractor_name(filename, used):

    base = re.sub(r"[^A-Za-z0-9]+", "_", Path(str(filename)).stem)
    base = base.strip("_").upper() or "ATTRACTOR"

    name = base
    counter = 2

    while name in used or name in RESERVED_NAMES:
        name = f"{base}_{counter}"
        counter += 1

    used.add(name)

    return name


def read_attractor_specs(cfg):

    task = cfg_value(cfg, "data.augmentation.task", "data.emoji.task")

    if task != "multi_attractor":
        raise ValueError(
            f"data.emoji.task is {task!r}, expected 'multi_attractor'"
        )

    pairs = plain(cfg_value(cfg, "data.augmentation.pairs", "data.emoji.pairs"))

    if not pairs:
        raise ValueError("data.emoji.pairs is empty or missing")

    specs = []
    used = set()

    for index, pair in enumerate(pairs):

        initial = pair.get("initial")
        target = pair.get("target")

        if isinstance(initial, str):
            initial = {"image": initial, "mode": "full"}

        if not isinstance(initial, dict) or not initial.get("image"):
            raise ValueError(f"pairs[{index}].initial has no image")

        if not target:
            raise ValueError(f"pairs[{index}].target is missing")

        specs.append(
            SimpleNamespace(
                index=index,
                name=attractor_name(target, used),
                target_file=str(target),
                initial=dict(initial),
            )
        )

    return specs


# ======================================================================
# MODELLE FINDEN UND PLANEN
# ======================================================================

def discover_bundles(root):

    bundles = []

    if not root.exists():
        return bundles

    for cfg_path in sorted(root.rglob("config.yaml")):

        bundle = cfg_path.parent

        if not (bundle / "model.eqx").is_file():
            continue

        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception:
            continue

        task = cfg_value(cfg, "data.augmentation.task", "data.emoji.task")

        if task == "multi_attractor":
            bundles.append(bundle)

    return bundles


def describe_bundle(bundle, emoji_dir):

    cfg_path = bundle / "config.yaml"
    model_path = bundle / "model.eqx"

    for path in (cfg_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")

    cfg = OmegaConf.load(cfg_path)
    padding = parse_padding_mode(cfg)
    specs = read_attractor_specs(cfg)

    needed = {
        filename
        for spec in specs
        for filename in (spec.target_file, spec.initial["image"])
    }

    missing = sorted(
        filename for filename in needed
        if not (emoji_dir / filename).is_file()
    )

    if missing:
        raise FileNotFoundError(
            f"emoji files not found in {emoji_dir}: {', '.join(missing)}"
        )

    return SimpleNamespace(
        bundle=bundle,
        cfg_path=cfg_path,
        model_path=model_path,
        cfg=cfg,
        padding=padding,
        family=str(cfg_value(cfg, "model.family", default="unknown")),
        specs=specs,
        n_attractors=len(specs),
        tag=None,
    )


def normalise_bundle_path(entry):

    path = Path(str(entry)).expanduser()

    if path.name in {"model.eqx", "config.yaml", "manifest.yaml"}:
        path = path.parent

    return path


def check_expected_models(plans):

    warnings = []

    if EXPECTED_FAMILY:
        for plan in plans:
            if plan.family != EXPECTED_FAMILY:
                warnings.append(
                    f"{plan.bundle} is family {plan.family!r}, "
                    f"expected {EXPECTED_FAMILY!r}"
                )

    if EXPECTED_MODELS:

        expected = Counter((n, p.upper()) for n, p in EXPECTED_MODELS)
        found = Counter((plan.n_attractors, plan.padding) for plan in plans)

        for combo in sorted(expected - found):
            warnings.append(
                f"missing: {combo[0]} attractors with {combo[1]} padding"
            )

        for combo in sorted(found - expected):
            warnings.append(
                f"unexpected or duplicate: {combo[0]} attractors with "
                f"{combo[1]} padding"
            )

    if warnings:

        print("\n**************************************************")
        print("WARNING: the models do not match EXPECTED_MODELS")
        print("**************************************************")

        for warning in warnings:
            print("  !", warning)

        print("The scan continues anyway. Check MODEL_BUNDLES if this is wrong.")

    return warnings


def plan_models(emoji_dir):

    placeholders = [
        index + 1
        for index, entry in enumerate(MODEL_BUNDLES)
        if str(entry).strip() in {"", PATH_PLACEHOLDER}
    ]

    if placeholders:
        raise RuntimeError(
            "MODEL_BUNDLES still contains placeholders (entries "
            f"{placeholders}). Replace every 'HIER_PFAD_EINTRAGEN' with the "
            "path of a bundle folder (the folder with model.eqx and "
            "config.yaml)."
        )

    explicit = bool(MODEL_BUNDLES)

    if explicit:
        bundles = [normalise_bundle_path(bundle) for bundle in MODEL_BUNDLES]
    else:
        print(f"MODEL_BUNDLES is empty -> searching {MODEL_SEARCH_ROOT}")
        bundles = discover_bundles(MODEL_SEARCH_ROOT)

    if not bundles:
        raise RuntimeError(
            "No model bundles found. Fill in MODEL_BUNDLES or check "
            "MODEL_SEARCH_ROOT."
        )

    plans = []
    problems = []

    for bundle in bundles:
        try:
            plans.append(describe_bundle(bundle, emoji_dir))
        except Exception as exc:
            problems.append(f"{bundle}: {type(exc).__name__}: {exc}")

    if problems:

        print("\nPROBLEMS WHILE READING BUNDLES:")

        for problem in problems:
            print("  -", problem)

        if explicit:
            raise RuntimeError(
                "Fix the bundles listed above before starting the scan."
            )

        print("These bundles are skipped.\n")

    plans.sort(
        key=lambda p: (p.n_attractors, p.padding, p.family, str(p.bundle))
    )

    used = set()

    for plan in plans:

        base = f"{plan.n_attractors}attr_{plan.padding.lower()}_{plan.family}"
        tag = base
        counter = 2

        while tag in used:
            tag = f"{base}_{counter}"
            counter += 1

        used.add(tag)
        plan.tag = tag

    check_expected_models(plans)

    return plans


# ======================================================================
# EMOJIS UND STARTPIXEL
# ======================================================================

def load_emoji(filename, emoji_dir, downsample, crop_square):

    dataset = load_emoji_sequence(
        [filename],
        impath_emojis=emoji_dir,
        downsample=downsample,
        crop_square=crop_square,
    )

    data = dataset.data if hasattr(dataset, "data") else dataset

    return np.asarray(data[0, 0], dtype=np.float32)


def build_initial_condition(initial, image):
    """
    Baut die Anfangsbedingung wie Experiments/emoji/config_helpers.py
    (_build_initial_condition) und gibt zusaetzlich die Position (y, x) des
    Startpixels im ungepaddeten Bild sowie eine Beschreibung zurueck.
    """

    mode = initial.get("mode", "full")
    channels, height, width = image.shape

    if mode == "full":

        position = ((height - 1) // 2, (width - 1) // 2)

        return image.copy(), position, "full image"

    if mode == "patch":

        size = int(initial.get("size", 12))

        if not 0 < size <= min(height, width):
            raise ValueError(
                f"patch size must be in [1, {min(height, width)}], got {size}"
            )

        top = (height - size) // 2
        left = (width - size) // 2

        result = np.zeros_like(image)
        result[:, top:top + size, left:left + size] = (
            image[:, top:top + size, left:left + size]
        )

        position = (top + (size - 1) // 2, left + (size - 1) // 2)

        return result, position, f"patch {size}x{size}"

    if mode == "pixel":

        channel = int(initial.get("channel", 0))
        value = float(initial.get("value", 1.0))

        if not 0 <= channel < channels:
            raise ValueError(
                f"pixel channel must be in [0, {channels - 1}], got {channel}"
            )

        result = np.zeros_like(image)
        result[channel, height // 2, width // 2] = value

        position = (height // 2, width // 2)

        return result, position, f"pixel (channel {channel} = {value})"

    raise ValueError(f"Unknown initial mode {mode!r}")


def load_attractors(specs, data, emoji_dir):

    cache = {}

    def get(filename):

        if filename not in cache:
            cache[filename] = load_emoji(
                filename, emoji_dir, data.downsample, data.crop_square
            )

        return cache[filename]

    attractors = []

    for spec in specs:

        target = get(spec.target_file)
        initial_image = get(spec.initial["image"])

        if initial_image.shape != target.shape:
            raise RuntimeError(
                f"{spec.name}: initial image {initial_image.shape} and "
                f"target {target.shape} have different shapes"
            )

        initial, seed_yx, description = build_initial_condition(
            spec.initial, initial_image
        )

        seed_pixel = initial[
            :data.data_channels, seed_yx[0], seed_yx[1]
        ].astype(np.float32)

        attractors.append(
            SimpleNamespace(
                index=spec.index,
                name=spec.name,
                target_file=spec.target_file,
                initial_file=spec.initial["image"],
                initial_description=description,
                initial_mode=spec.initial.get("mode", "full"),
                target=target,
                seed_yx=seed_yx,
                seed_pixel=seed_pixel,
                colour=None,
            )
        )

    shapes = {a.target.shape for a in attractors}

    if len(shapes) != 1:
        raise RuntimeError(f"Attractor targets have different shapes: {shapes}")

    return attractors


def complementary_pixel(pixel):

    result = np.array(pixel, dtype=np.float32, copy=True)
    result[:3] = 1.0 - result[:3]

    return result


def make_pixel(data_channels, r=0.0, g=0.0, b=0.0, a=None):

    pixel = np.zeros((data_channels,), dtype=np.float32)

    for channel, value in enumerate((r, g, b)):
        if channel < data_channels:
            pixel[channel] = value

    if a is not None and data_channels >= 4:
        pixel[3] = a

    return pixel


def pixel_to_255(pixel):

    return np.round(np.asarray(pixel) * 255).astype(int).tolist()


# ======================================================================
# MODELL LADEN UND SIMULIEREN
# ======================================================================

def load_model(cfg, model_path):

    result = build_model(cfg.model, key=jr.PRNGKey(0))
    model = result[0] if isinstance(result, tuple) else result

    return model.load(model_path)


def make_step_keys(steps, seed):
    """Dieselbe Schluesselfolge wie im Original: key, subkey = split(key)."""

    key = jr.PRNGKey(seed)
    subkeys = []

    for _ in range(steps):
        key, subkey = jr.split(key)
        subkeys.append(subkey)

    return jnp.stack(subkeys)


def shift_for_analysis_jax(state, dy, dx, circular):
    """
    JAX-Gegenstueck zu align_for_analysis: verschiebt den Zustand so, dass
    der Seed an der Trainingsposition liegt. dy/dx duerfen getracet sein.
    """

    rolled = jnp.roll(state, shift=(dy, dx), axis=(-2, -1))

    if circular:
        return rolled

    # Ohne Wrap-around: alles, was von der anderen Seite hereingerollt ist,
    # wieder auf 0 setzen (wie shift_no_wrap).
    height = state.shape[-2]
    width = state.shape[-1]

    rows = jnp.arange(height)
    cols = jnp.arange(width)

    keep_rows = (rows - dy >= 0) & (rows - dy < height)
    keep_cols = (cols - dx >= 0) & (cols - dx < width)

    return rolled * (keep_rows[:, None] & keep_cols[None, :])


def classify_state_jax(state, dy, dx, targets, data_channels, circular,
                       n_names):
    """
    Dieselbe Entscheidung wie classify_state, aber in JAX und ohne Namen:
    gibt den Index der Klasse zurueck.

        0 .. n_names-1   der jeweilige Attraktor
        n_names          INTERMEDIATE
        n_names + 1      NO_VISIBLE_PATTERN
    """

    aligned = shift_for_analysis_jax(state, dy, dx, circular)

    dc = data_channels
    visible = aligned[:dc]

    mse = jnp.mean((visible[None] - targets[:, :dc]) ** 2, axis=(1, 2, 3))

    per_pixel_error = jnp.mean(
        (visible[None, :3] - targets[:, :3]) ** 2, axis=1
    )

    if dc >= 4:
        active_mask = visible[3] > ACTIVE_ALPHA_THRESHOLD
    else:
        active_mask = jnp.max(jnp.abs(visible[:3]), axis=0) > 0.10

    active_pixels = jnp.sum(active_mask)

    if n_names == 1:
        fractions = jnp.where(active_pixels > 0, 1.0, 0.0)[None]

    else:
        # Ein Pixel zaehlt nur fuer einen Attraktor, wenn er diesem
        # eindeutig am naechsten ist (striktes "<" wie in classify_state).
        ordered = jnp.sort(per_pixel_error, axis=0)
        unique = ordered[0] < ordered[1]
        nearest = jnp.argmin(per_pixel_error, axis=0)
        counted = unique & active_mask

        counts = jnp.stack([
            jnp.sum(counted & (nearest == k)) for k in range(n_names)
        ])
        fractions = counts / jnp.maximum(active_pixels, 1)

    best_match = jnp.argmin(mse)
    mixed_count = jnp.sum(fractions >= MIXED_MIN_FRACTION)

    label = jnp.where(mixed_count >= 2, n_names, best_match)

    return jnp.where(active_pixels == 0, n_names + 1, label).astype(jnp.int32)


def state_to_rgba_jax(state):
    """JAX-Gegenstueck zu state_to_rgba: (C, H, W) -> (4, H, W) uint8."""

    return jnp.round(jnp.clip(state[:4], 0.0, 1.0) * 255).astype(jnp.uint8)


@eqx.filter_jit
def rollout_batch(model, states, step_keys, data_channels, targets, shifts,
                  circular, n_names, measure_labels, record_frames):
    """
    Rollout mit Konvergenzmessung.

    Rueckgabe: (Endzustand,
                Aenderung je Schritt ueber die sichtbaren Kanaele,
                Aenderung je Schritt ueber alle Kanaele,
                Klassen-Index je Schritt,
                Bilder je Schritt oder None).

    Die Reihen haben die Form (BATCH, STEPS).
    change[t] = mean((state nach Schritt t+1 - state nach Schritt t)**2).
    Sie fallen im selben lax.scan mit ab und kosten deshalb praktisch keine
    zusaetzliche Rechenzeit.

    record_frames=True speichert zusaetzlich nach jedem Schritt das sichtbare
    RGBA-Bild als uint8: (BATCH, STEPS, 4, H, W). Das kostet Speicher (bei
    FRAME_BATCH_SIZE = 32 und 140 x 140 rund 320 MB), deshalb nur fuer die
    Laeufe, von denen ein Entwicklungs-GIF geschrieben wird.

    shifts: (BATCH, 2) mit (dy, dx) je Lauf, also der Verschiebung, die den
    Seed auf die Trainingsposition bringt.
    """

    def single(state, shift):

        dy = shift[0]
        dx = shift[1]

        def body(current, key):

            following = model(current, key=key)
            difference = following - current

            if measure_labels:
                label = classify_state_jax(
                    following, dy, dx, targets, data_channels, circular,
                    n_names,
                )
            else:
                label = jnp.int32(-1)

            outputs = (
                jnp.mean(difference[:data_channels] ** 2),
                jnp.mean(difference ** 2),
                label,
            )

            if record_frames:
                outputs = outputs + (state_to_rgba_jax(following),)

            return following, outputs

        final, outputs = jax.lax.scan(body, state, step_keys)

        if record_frames:
            visible_change, all_change, labels, frames = outputs
        else:
            visible_change, all_change, labels = outputs
            frames = None

        return final, visible_change, all_change, labels, frames

    return jax.vmap(single)(states, shifts)


@eqx.filter_jit
def distance_to_final_batch(model, states, step_keys, finals, data_channels):
    """
    Zweiter Durchlauf (nur wenn MEASURE_DISTANCE_TO_FINAL gesetzt ist):
    misst je Schritt den Abstand zum bereits bekannten Endzustand.

    Rueckgabe: (BATCH_SIZE, STEPS) mit
    distance[t] = mean((state nach Schritt t+1 - Endzustand)**2)
    ueber die sichtbaren Kanaele.
    """

    def single(state, final):

        def body(current, key):

            following = model(current, key=key)
            difference = following[:data_channels] - final[:data_channels]

            return following, jnp.mean(difference ** 2)

        _, distances = jax.lax.scan(body, state, step_keys)

        return distances

    return jax.vmap(single)(states, finals)


def smooth_series(values, window):
    """Zentrierter gleitender Median; die Raender werden fortgeschrieben."""

    values = np.asarray(values, dtype=np.float64)

    if window <= 1 or values.size <= 1:
        return values

    window = int(min(window, values.size))
    left = window // 2
    right = window - 1 - left

    padded = np.pad(values, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)

    return np.median(windows, axis=-1)


def label_sequence_rle(labels, class_names, max_segments=24):
    """
    Die Klassenreihe eines Laufs als Lauflaengen-Kodierung, z.B.

        NO_VISIBLE_PATTERN@1-2, INTERMEDIATE@3-11, CRAB@12-97,
        MICROBE@98, CRAB@99-128

    Bei wenigen Wechseln sind das ein paar Zeichen pro Lauf. Damit ist in
    der CSV direkt sichtbar, WOHIN die Klassifikation zwischendurch kippt -
    ohne diese Spalte sieht man nur, dass sie irgendwann abwich.
    """

    labels = np.asarray(labels)

    if labels.size == 0:
        return ""

    def name_of(index):
        index = int(index)
        return (
            class_names[index]
            if 0 <= index < len(class_names) else f"?{index}"
        )

    segments = []
    start = 0

    for position in range(1, labels.size + 1):

        if position == labels.size or labels[position] != labels[start]:

            first = start + 1          # 1-basierte Schrittzaehlung
            last = position

            segments.append(
                f"{name_of(labels[start])}@{first}"
                + ("" if first == last else f"-{last}")
            )
            start = position

    if len(segments) > max_segments:
        kept = max_segments - 1
        segments = segments[:kept] + [f"...(+{len(segments) - kept})"]

    return ", ".join(segments)


def last_difference_step(labels, reference):
    """
    Letzter Schritt, an dem die Klassifikation noch von `reference` abwich.

    labels[i] gehoert zu Schritt i + 1. Rueckgabe wie bei stable_from_step:
    0 heisst "schon nach dem ersten Schritt dort", len(labels) heisst "bis
    zum Ende nicht dauerhaft dort".
    """

    labels = np.asarray(labels)
    differing = np.nonzero(labels != reference)[0]

    if differing.size == 0:
        return 0

    return int(differing[-1]) + 1


def stable_from_step(series, threshold):
    """
    Letzter Schritt, bei dem die Reihe noch ueber der Schwelle liegt.

    series[i] gehoert zu Schritt i + 1 (1-basierte Schrittzaehlung).
    Rueckgabe:
      0             -> schon vom ersten Schritt an unter der Schwelle
      len(series)   -> bis zum Ende nicht stabil geworden
    """

    values = np.asarray(series, dtype=np.float64)
    above = np.nonzero(values > threshold)[0]

    if above.size == 0:
        return 0

    return int(above[-1]) + 1


def convergence_metrics(visible_change, all_change, distances=None,
                        seconds_per_run=None, labels=None, class_names=None):
    """Macht aus den Reihen eines Laufs die Ergebnis-Spalten."""

    visible_change = np.asarray(visible_change, dtype=np.float64)
    all_change = np.asarray(all_change, dtype=np.float64)

    raw_series = (
        visible_change if CONVERGENCE_USE_VISIBLE_CHANNELS else all_change
    )
    total_steps = int(raw_series.size)

    # Geglaettet, damit das Zappeln durch fire_rate < 1 nicht als Bewegung
    # zaehlt. Alle Schwellen beziehen sich auf die geglaettete Reihe.
    series = smooth_series(raw_series, CONVERGENCE_SMOOTH_WINDOW)

    tail = series[-min(CONVERGENCE_TAIL_STEPS, total_steps):]
    residual = float(np.median(tail))
    peak = float(np.max(series)) if total_steps else 0.0

    # Ist der Lauf ueberhaupt zur Ruhe gekommen? Nur wenn die Restaenderung
    # am Ende klein gegenueber der groessten Aenderung des Laufs ist. Sonst
    # schwingt das Muster noch (oder laeuft davon) und "stabil" waere falsch.
    settled = (
        residual <= CONVERGENCE_SETTLED_FRACTION * peak
        or residual <= CONVERGENCE_ABS_TOL
    )

    threshold = max(CONVERGENCE_ABS_TOL, CONVERGENCE_REL_TOL * residual)

    steps = stable_from_step(series, threshold) if settled else total_steps
    steps_abs = stable_from_step(series, CONVERGENCE_ABS_TOL)

    after_stable = series[steps:]

    result = {
        "steps_to_stable": steps,
        "stable_after_fraction": (
            float(steps) / total_steps if total_steps else ""
        ),
        "converged_before_end": bool(settled and steps < total_steps),
        "steps_to_stable_abs_tol": steps_abs,
        "change_threshold": threshold,
        "residual_change_per_step": residual,
        "peak_change_per_step": peak,
        "residual_over_peak": (residual / peak) if peak > 0 else 0.0,
        "final_change_per_step": float(raw_series[-1]) if total_steps else "",
        "max_change_after_stable": (
            float(np.max(after_stable)) if after_stable.size else ""
        ),
        "change_smooth_window": CONVERGENCE_SMOOTH_WINDOW,
        "mean_change_all_channels": float(np.mean(all_change)),
        "total_steps": total_steps,
    }

    if labels is not None and class_names is not None:

        labels = np.asarray(labels)
        final_label = int(labels[-1])

        steps_to_attractor = last_difference_step(labels, final_label)
        switches = int(np.sum(labels[1:] != labels[:-1]))

        final_name = (
            class_names[final_label]
            if 0 <= final_label < len(class_names) else "?"
        )

        result["steps_to_attractor"] = steps_to_attractor
        result["attractor_after_fraction"] = (
            float(steps_to_attractor) / total_steps if total_steps else ""
        )
        result["attractor_locked_before_end"] = (
            steps_to_attractor < total_steps
        )
        result["attractor_label_final"] = final_name
        result["attractor_label_switches"] = switches
        result["label_sequence"] = label_sequence_rle(labels, class_names)

        # Strengeres Mass: ab wann ist der Zustand einem KONKRETEN Muster
        # zugeordnet und faellt bis zum Ende nicht mehr in INTERMEDIATE oder
        # NO_VISIBLE_PATTERN zurueck (und auch nicht auf einen anderen
        # Attraktor)?
        #
        # Endet der Lauf nicht auf einem echten Attraktor, gibt es keinen
        # solchen Schritt - die Spalte bleibt dann leer, statt eine Zahl zu
        # liefern, die in Wahrheit nur sagt, wann sich der Mischzustand
        # eingependelt hat.
        #
        # Steht am Ende ein echter Attraktor, ist die Menge der verletzenden
        # Schritte dieselbe wie oben (alles, was nicht die End-Klasse war),
        # der Wert also identisch zu steps_to_attractor.
        reaches_pure = final_name not in RESERVED_NAMES

        result["reaches_pure_attractor"] = reaches_pure
        result["steps_to_pure_attractor"] = (
            last_difference_step(labels, final_label) if reaches_pure else ""
        )
        result["pure_attractor_after_fraction"] = (
            float(result["steps_to_pure_attractor"]) / total_steps
            if reaches_pure and total_steps else ""
        )

    if seconds_per_run is not None:

        result["rollout_seconds_per_run"] = float(seconds_per_run)
        result["estimated_seconds_to_stable"] = (
            float(seconds_per_run) * steps / total_steps
            if total_steps else ""
        )

        if "steps_to_attractor" in result:
            result["estimated_seconds_to_attractor"] = (
                float(seconds_per_run) * result["steps_to_attractor"]
                / total_steps if total_steps else ""
            )

    if distances is not None:

        distances = np.asarray(distances, dtype=np.float64)

        # Der letzte Eintrag ist per Definition 0 (Abstand zu sich selbst),
        # deshalb geht er nicht ins Rauschniveau ein.
        distance_tail = distances[
            max(total_steps - CONVERGENCE_TAIL_STEPS, 0):max(total_steps - 1, 1)
        ]
        distance_residual = (
            float(np.median(distance_tail)) if distance_tail.size else 0.0
        )
        distance_peak = float(np.max(distances)) if distances.size else 0.0
        distance_threshold = max(
            CONVERGENCE_DISTANCE_TOL, CONVERGENCE_REL_TOL * distance_residual
        )

        distance_settled = (
            distance_residual <= CONVERGENCE_SETTLED_FRACTION * distance_peak
            or distance_residual <= CONVERGENCE_DISTANCE_TOL
        )

        result["steps_to_final_pattern"] = (
            stable_from_step(distances, distance_threshold)
            if distance_settled else total_steps
        )
        result["distance_threshold"] = distance_threshold
        result["residual_distance_to_final"] = distance_residual

    return result


def empty_convergence_summary():

    return {
        "runs": 0,
        "mean_steps_to_stable": "",
        "median_steps_to_stable": "",
        "min_steps_to_stable": "",
        "max_steps_to_stable": "",
        "runs_not_settled": 0,
        "mean_estimated_seconds_to_stable": "",
        "mean_steps_to_attractor": "",
        "median_steps_to_attractor": "",
        "max_steps_to_attractor": "",
        "runs_attractor_not_locked": 0,
        "runs_reaching_pure_attractor": 0,
        "fraction_reaching_pure_attractor": "",
        "mean_steps_to_pure_attractor": "",
        "median_steps_to_pure_attractor": "",
        "max_steps_to_pure_attractor": "",
    }


def summarise_convergence(rows):
    """Mittelwerte der Konvergenz ueber beliebig viele Ergebniszeilen."""

    rows = [row for row in rows if "steps_to_stable" in row]

    if not rows:
        return empty_convergence_summary()

    steps = np.asarray(
        [row["steps_to_stable"] for row in rows], dtype=np.float64
    )
    seconds = [
        row["estimated_seconds_to_stable"]
        for row in rows
        if isinstance(row.get("estimated_seconds_to_stable"), float)
    ]

    summary = {
        "runs": len(rows),
        "mean_steps_to_stable": float(np.mean(steps)),
        "median_steps_to_stable": float(np.median(steps)),
        "min_steps_to_stable": int(np.min(steps)),
        "max_steps_to_stable": int(np.max(steps)),
        "runs_not_settled": sum(
            1 for row in rows if not row["converged_before_end"]
        ),
        "mean_estimated_seconds_to_stable": (
            float(np.mean(seconds)) if seconds else ""
        ),
    }

    attractor_rows = [row for row in rows if "steps_to_attractor" in row]

    if attractor_rows:

        attractor_steps = np.asarray(
            [row["steps_to_attractor"] for row in attractor_rows],
            dtype=np.float64,
        )

        summary.update({
            "mean_steps_to_attractor": float(np.mean(attractor_steps)),
            "median_steps_to_attractor": float(np.median(attractor_steps)),
            "max_steps_to_attractor": int(np.max(attractor_steps)),
            "runs_attractor_not_locked": sum(
                1 for row in attractor_rows
                if not row["attractor_locked_before_end"]
            ),
        })

        # Nur die Laeufe, die ueberhaupt auf einem echten Attraktor landen.
        # Die Quote gehoert zwingend daneben: der Mittelwert bezieht sich
        # sonst unbemerkt nur auf die erfolgreichen Laeufe.
        pure_steps = [
            row["steps_to_pure_attractor"]
            for row in attractor_rows
            if row.get("reaches_pure_attractor")
        ]

        summary["runs_reaching_pure_attractor"] = len(pure_steps)
        summary["fraction_reaching_pure_attractor"] = (
            len(pure_steps) / len(attractor_rows)
        )

        if pure_steps:

            pure_steps = np.asarray(pure_steps, dtype=np.float64)

            summary.update({
                "mean_steps_to_pure_attractor": float(np.mean(pure_steps)),
                "median_steps_to_pure_attractor": float(np.median(pure_steps)),
                "max_steps_to_pure_attractor": int(np.max(pure_steps)),
            })

        else:
            summary.update({
                "mean_steps_to_pure_attractor": "",
                "median_steps_to_pure_attractor": "",
                "max_steps_to_pure_attractor": "",
            })

    else:
        summary.update({
            key: value
            for key, value in empty_convergence_summary().items()
            if "attractor" in key
        })

    return summary


def split_by_frame_need(needs_frames):
    """
    Zerlegt die Laufliste in zusammenhaengende Abschnitte mit gleichem
    Bedarf an Einzelbildern: [(start, ende, braucht_bilder), ...].
    Die Reihenfolge bleibt erhalten.
    """

    groups = []
    start = 0

    for index in range(1, len(needs_frames) + 1):

        if index == len(needs_frames) or \
                needs_frames[index] != needs_frames[start]:
            groups.append((start, index, bool(needs_frames[start])))
            start = index

    return groups


def iterate_rollouts(ctx, seeds, needs_frames=None):
    """
    seeds: Liste von (pixel, seed_x, seed_y).
    needs_frames: optional, Liste von bool je Lauf. Fuer Laeufe mit True
    werden alle Zwischenbilder aufgezeichnet (fuer das Entwicklungs-GIF).

    Liefert (index, Endzustand, Konvergenz-Spalten, extras) in der
    Reihenfolge der Liste. extras hat
        labels  Klassen-Index je Schritt (STEPS,)
        frames  (STEPS + 1, 4, H, W) uint8 inkl. Seed-Bild, oder None

    Laeufe ohne Bildbedarf laufen in Bloecken zu BATCH_SIZE, Laeufe mit
    Bildbedarf in Bloecken zu FRAME_BATCH_SIZE (weniger Speicher). Jeder
    Block wird auf seine volle Groesse mit Nullen aufgefuellt, damit jede
    Variante nur einmal kompiliert wird.
    """

    if needs_frames is None:
        needs_frames = [False] * len(seeds)

    for group_start, group_end, record in split_by_frame_need(needs_frames):

        batch_size = FRAME_BATCH_SIZE if record else BATCH_SIZE

        for start in range(group_start, group_end, batch_size):

            stop = min(start + batch_size, group_end)
            yield from _rollout_block(
                ctx, seeds, start, stop, batch_size, record
            )


def _rollout_block(ctx, seeds, start, stop, batch_size, record):
    """Ein Block von Laeufen: seeds[start:stop], aufgefuellt auf batch_size."""

    chunk = seeds[start:stop]
    shape = (batch_size, ctx.n_channels, ctx.canvas_H, ctx.canvas_W)

    states = np.zeros(shape, dtype=np.float32)

    for offset, (pixel, seed_x, seed_y) in enumerate(chunk):
        states[offset, :ctx.data_channels, seed_y, seed_x] = pixel

    # (dy, dx) je Lauf: bringt den Seed auf die Trainingsposition.
    shifts = np.zeros((batch_size, 2), dtype=np.int32)

    for offset, (pixel, seed_x, seed_y) in enumerate(chunk):
        shifts[offset] = (ctx.training_y - seed_y, ctx.training_x - seed_x)

    initial_states = jnp.asarray(states)
    batch_started = time.time()

    final, visible_change, all_change, labels, frames = rollout_batch(
        ctx.model,
        initial_states,
        ctx.step_keys,
        ctx.data_channels,
        jnp.asarray(ctx.targets_full),
        jnp.asarray(shifts),
        ctx.circular,
        len(ctx.names),
        MEASURE_ATTRACTOR_STEPS,
        record,
    )

    final = np.asarray(jax.device_get(final))
    visible_change = np.asarray(jax.device_get(visible_change))
    all_change = np.asarray(jax.device_get(all_change))
    labels = np.asarray(jax.device_get(labels))

    if record:
        frames = np.asarray(jax.device_get(frames))
        # Bild 0 ist der Seed selbst, danach Schritt 1 .. STEPS.
        seed_frames = np.round(
            np.clip(states[:, :4], 0.0, 1.0) * 255
        ).astype(np.uint8)

    if MEASURE_DISTANCE_TO_FINAL:

        distances = np.asarray(jax.device_get(distance_to_final_batch(
            ctx.model,
            initial_states,
            ctx.step_keys,
            jnp.asarray(final),
            ctx.data_channels,
        )))

    else:
        distances = None

    # Nur ein Richtwert: die gemessene Zeit des Blocks geteilt durch die
    # Anzahl der tatsaechlich benutzten Laeufe darin.
    seconds_per_run = (time.time() - batch_started) / max(len(chunk), 1)

    for offset in range(len(chunk)):

        dynamics = convergence_metrics(
            visible_change[offset],
            all_change[offset],
            distances[offset] if distances is not None else None,
            seconds_per_run,
            labels[offset] if MEASURE_ATTRACTOR_STEPS else None,
            ctx.class_order if MEASURE_ATTRACTOR_STEPS else None,
        )

        extras = SimpleNamespace(
            labels=labels[offset].copy(),
            frames=(
                np.concatenate(
                    [seed_frames[offset][None], frames[offset]], axis=0
                )
                if record else None
            ),
        )

        yield start + offset, final[offset], dynamics, extras


# ======================================================================
# ANALYSE
# ======================================================================

def shift_no_wrap(arr, dy, dx):

    arr = np.asarray(arr)
    result = np.zeros_like(arr)

    height = arr.shape[-2]
    width = arr.shape[-1]

    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    src_y0 = max(0, -dy)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_x0 = max(0, -dx)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        result[..., dst_y0:dst_y1, dst_x0:dst_x1] = (
            arr[..., src_y0:src_y1, src_x0:src_x1]
        )

    return result


def align_for_analysis(ctx, state, seed_x, seed_y):
    """
    Verschiebt den Zustand so, dass der Seed an der Trainingsposition liegt.
    CIRCULAR: mit Wrap-around (nichts geht verloren).
    Sonst: ohne Wrap-around, herausgeschobene Teile gehen verloren.
    """

    dx = ctx.training_x - seed_x
    dy = ctx.training_y - seed_y

    if ctx.circular:
        return np.roll(state, shift=(dy, dx), axis=(-2, -1))

    return shift_no_wrap(state, dy, dx)


def classify_state(ctx, raw_state, seed_x, seed_y):

    aligned = align_for_analysis(ctx, raw_state, seed_x, seed_y)

    dc = ctx.data_channels
    names = ctx.names
    n = len(names)

    visible = aligned[:dc]
    targets = ctx.targets_full

    mse = np.mean(
        (visible[None] - targets[:, :dc]) ** 2,
        axis=(1, 2, 3),
    )

    per_pixel_error = np.mean(
        (visible[None, :3] - targets[:, :3]) ** 2,
        axis=1,
    )

    if dc >= 4:
        active_mask = visible[3] > ACTIVE_ALPHA_THRESHOLD
    else:
        active_mask = np.max(np.abs(visible[:3]), axis=0) > 0.10

    active_pixels = int(np.sum(active_mask))

    fractions = np.zeros(n, dtype=np.float64)

    if active_pixels > 0:

        if n == 1:
            fractions[0] = 1.0

        else:
            # Ein Pixel zaehlt nur fuer einen Attraktor, wenn er diesem
            # eindeutig am naechsten ist (wie das strikte "<" im Original).
            ordered = np.sort(per_pixel_error, axis=0)
            unique = ordered[0] < ordered[1]
            nearest = np.argmin(per_pixel_error, axis=0)
            counted = unique & active_mask

            for k in range(n):
                fractions[k] = np.sum(counted & (nearest == k)) / active_pixels

    best_match = names[int(np.argmin(mse))]

    mixed = [
        names[k] for k in range(n)
        if fractions[k] >= MIXED_MIN_FRACTION
    ]

    if active_pixels == 0:
        classification = NO_PATTERN
    elif len(mixed) >= 2:
        classification = INTERMEDIATE
    else:
        classification = best_match

    result = {"classification": classification}

    for k, name in enumerate(names):
        result[f"mse_{name.lower()}"] = float(mse[k])

    for k, name in enumerate(names):
        result[f"{name.lower()}_fraction"] = float(fractions[k])

    result["best_match"] = best_match
    result["mixed_components"] = (
        "+".join(mixed) if classification == INTERMEDIATE else ""
    )
    result["active_pixels"] = active_pixels
    result["aligned_state"] = aligned

    return result


def compare_to_center(raw_state, aligned_state, center_state, dc):

    return {
        "raw_visible_mse_vs_center": float(np.mean(
            (raw_state[:dc] - center_state[:dc]) ** 2
        )),
        "aligned_visible_mse_vs_center": float(np.mean(
            (aligned_state[:dc] - center_state[:dc]) ** 2
        )),
        "raw_all_channels_mse_vs_center": float(np.mean(
            (raw_state - center_state) ** 2
        )),
        "aligned_all_channels_mse_vs_center": float(np.mean(
            (aligned_state - center_state) ** 2
        )),
    }


def state_to_rgba(state):

    image = np.moveaxis(np.asarray(state[:4]), 0, -1)
    image = np.clip(image, 0.0, 1.0)

    return np.round(image * 255).astype(np.uint8)


def save_rgba(array, path):

    Image.fromarray(array).save(path)


# ======================================================================
# EVOLUTIONS-GIFS
# ======================================================================

def evolution_positions(ctx):
    """Namen der Positionen, fuer die Entwicklungs-GIFs geschrieben werden."""

    names = [name for name, _, _ in ctx.positions]

    if EVOLUTION_GIF_POSITIONS == "all":
        return set(names)

    unknown = set(EVOLUTION_GIF_POSITIONS) - set(names)

    if unknown:
        raise ValueError(
            f"EVOLUTION_GIF_POSITIONS enthaelt unbekannte Positionen: "
            f"{sorted(unknown)}; moeglich sind {names}"
        )

    return set(EVOLUTION_GIF_POSITIONS)


def reconcile_with_classification(dynamics, labels, classification,
                                  class_order):
    """
    Die Klasse je Schritt kommt aus JAX (float32), die offizielle
    End-Klassifikation aus numpy (float64). Liegt ein Lauf exakt auf einer
    Entscheidungsgrenze (z.B. ein Anteil von genau 0.20), koennen beide beim
    letzten Schritt verschieden entscheiden. In dem Fall gilt die
    numpy-Klassifikation, und label_boundary_tie markiert den Lauf.
    """

    if "attractor_label_final" not in dynamics:
        return dynamics

    tie = dynamics["attractor_label_final"] != classification
    dynamics["label_boundary_tie"] = tie

    if not tie:
        return dynamics

    total_steps = len(labels)
    reference = class_order.index(classification)
    steps = last_difference_step(labels, reference)
    reaches = classification not in RESERVED_NAMES

    dynamics["attractor_label_final"] = classification
    dynamics["steps_to_attractor"] = steps
    dynamics["attractor_after_fraction"] = steps / total_steps
    dynamics["attractor_locked_before_end"] = steps < total_steps
    dynamics["reaches_pure_attractor"] = reaches
    dynamics["steps_to_pure_attractor"] = steps if reaches else ""
    dynamics["pure_attractor_after_fraction"] = (
        steps / total_steps if reaches else ""
    )

    return dynamics


def evolution_frames(frames, labels, class_order, header, dynamics,
                     classification):
    """
    Baut die beschrifteten GIF-Bilder eines Laufs.

    frames: (STEPS + 1, 4, H, W) uint8, Bild 0 ist der Seed.
    Beschriftung je Bild:
        Zeile 1  header (z.B. "alpha=0.40 | center")
        Zeile 2  "step t/STEPS"
        Zeile 3  Klasse in diesem Schritt ("seed" bei t = 0)
        Zeile 4  ab dem Schritt, ab dem das Bild einem Attraktor fest
                 zugeordnet bleibt: "attractor at N/STEPS"; im letzten Bild
                 bei Laeufen ohne reinen Attraktor: "no pure attractor"
    """

    total_steps = frames.shape[0] - 1
    reaches = bool(dynamics.get("reaches_pure_attractor"))
    pure = dynamics.get("steps_to_pure_attractor")
    lock_from = (int(pure) + 1) if reaches and pure != "" else None

    indices = list(range(0, total_steps + 1, max(EVOLUTION_FRAME_STRIDE, 1)))

    if indices[-1] != total_steps:
        indices.append(total_steps)       # das Endbild immer mitnehmen

    images = []

    for t in indices:

        # Wie beim bisherigen GIF: RGBA -> RGB, der Alpha-Kanal faellt weg.
        rgb = np.ascontiguousarray(np.moveaxis(frames[t, :3], 0, -1))
        image = Image.fromarray(rgb)

        if EVOLUTION_UPSCALE > 1:
            image = image.resize(
                (image.width * EVOLUTION_UPSCALE,
                 image.height * EVOLUTION_UPSCALE),
                resample=Image.Resampling.NEAREST,
            )

        if t == 0:
            state_name = "seed"
        elif t == total_steps:
            state_name = classification    # offizielle End-Klassifikation
        else:
            index = int(labels[t - 1])
            state_name = (
                class_order[index] if 0 <= index < len(class_order) else "?"
            )

        draw = ImageDraw.Draw(image)
        draw.text((5, 5), header)
        draw.text((5, 20), f"step {t}/{total_steps}")
        draw.text((5, 35), state_name)

        if lock_from is not None and t >= lock_from:
            draw.text((5, 50), f"attractor at {pure}/{total_steps}")
        elif t == total_steps and not reaches:
            draw.text((5, 50), "no pure attractor")

        images.append(image)

    durations = [EVOLUTION_FRAME_MS] * len(images)
    durations[0] = EVOLUTION_SEED_HOLD_MS
    durations[-1] = EVOLUTION_END_HOLD_MS

    return images, durations


def write_evolution_gif(path, frames, labels, class_order, header, dynamics,
                        classification):

    images, durations = evolution_frames(
        frames, labels, class_order, header, dynamics, classification
    )

    if EVOLUTION_FAST_QUANTIZE:
        # Eigene, schnelle Farbreduktion auf 256 Farben je Bild. Ohne das
        # macht Pillow es beim Speichern mit einem langsameren Verfahren.
        images = [
            image.quantize(256, method=Image.Quantize.FASTOCTREE)
            for image in images
        ]

    path.parent.mkdir(parents=True, exist_ok=True)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )

    return path


def gif_selected_by_class(classification):
    """Bekommt ein Lauf mit dieser End-Klasse ein Entwicklungs-GIF?"""

    return EVOLUTION_GIF_CLASSES is None or \
        classification in EVOLUTION_GIF_CLASSES


def record_selected_gifs(ctx, seeds, selected, rows_by_index, describe):
    """
    Zweiter Durchgang: rechnet nur die ausgewaehlten Laeufe noch einmal,
    diesmal mit Bildaufzeichnung, und schreibt deren Entwicklungs-GIFs.

    seeds         alle Laeufe des Scans, (pixel, seed_x, seed_y)
    selected      Indizes in seeds, die ein GIF bekommen
    rows_by_index Ergebniszeile je Index aus dem ersten Durchgang; wird um
                  evolution_gif und evolution_gif_replay_matches ergaenzt
    describe(i)   -> (gif_pfad, kopfzeile)

    Die Laeufe sind deterministisch. Zur Sicherheit wird je Lauf geprueft,
    dass der zweite Durchgang dieselbe Klassenreihe und dieselbe
    End-Klassifikation liefert. Rueckgabe: Anzahl Abweichungen.
    """

    if not selected:
        return 0

    replay_seeds = [seeds[i] for i in selected]
    mismatches = 0

    for k, raw_state, dynamics, extras in iterate_rollouts(
        ctx, replay_seeds, [True] * len(replay_seeds)
    ):

        i = selected[k]
        row = rows_by_index[i]
        _, seed_x, seed_y = seeds[i]

        replay_class = classify_state(ctx, raw_state, seed_x, seed_y)[
            "classification"
        ]
        matches = (
            replay_class == row["classification"]
            and dynamics.get("label_sequence") == row.get("label_sequence")
        )

        if not matches:
            mismatches += 1

        path, header = describe(i)

        ctx.gif_writer.submit(
            path,
            extras.frames,
            extras.labels,
            ctx.class_order,
            header,
            {
                key: row[key]
                for key in ("reaches_pure_attractor", "steps_to_pure_attractor")
                if key in row
            },
            row["classification"],
        )

        row["evolution_gif"] = str(path.relative_to(ctx.output_dir))
        row["evolution_gif_replay_matches"] = matches

    ctx.replay_mismatches += mismatches

    return mismatches


class GifWriter:
    """
    Schreibt GIFs in einem Thread-Pool, waehrend die GPU schon den naechsten
    Block rechnet. Die Zahl gleichzeitig offener Auftraege ist begrenzt,
    damit die Bilddaten nicht unbegrenzt im Speicher liegen bleiben.
    Fehler beim Schreiben werden gesammelt und am Ende gemeldet, statt den
    ganzen Lauf abzubrechen.
    """

    def __init__(self, threads):

        self.threads = max(int(threads), 1)
        self.pool = ThreadPoolExecutor(max_workers=self.threads)
        self.pending = []
        self.written = 0
        self.failures = []

    def submit(self, path, *args):

        self.pending.append((path, self.pool.submit(
            write_evolution_gif, path, *args
        )))

        # Begrenzen: aeltesten Auftrag abwarten, wenn zu viele offen sind.
        while len(self.pending) > 4 * self.threads:
            self._collect(self.pending.pop(0))

    def _collect(self, item):

        path, future = item

        try:
            future.result()
            self.written += 1
        except Exception as exc:
            self.failures.append(f"{path}: {type(exc).__name__}: {exc}")

    def finish(self):
        """Wartet alle offenen Auftraege ab. Rueckgabe: (geschrieben,
        Fehler) seit dem letzten Aufruf."""

        while self.pending:
            self._collect(self.pending.pop(0))

        written, failures = self.written, self.failures
        self.written, self.failures = 0, []

        return written, failures

    def close(self):

        self.finish()
        self.pool.shutdown(wait=True)


def pixel_columns(pixel, dc):

    values = np.asarray(pixel)

    return {
        "R": float(values[0]) if dc >= 1 else "",
        "G": float(values[1]) if dc >= 2 else "",
        "B": float(values[2]) if dc >= 3 else "",
        "A": float(values[3]) if dc >= 4 else "",
    }


# ======================================================================
# HILFSFUNKTIONEN FUER AUSGABEN
# ======================================================================

def write_csv(path, rows):

    if not rows:
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def class_count_columns(ctx, classifications):

    counts = {}

    for name in ctx.class_colours:
        counts[f"{name.lower()}_count"] = sum(
            1 for c in classifications if c == name
        )

    return counts


def describe_transitions(rows):

    if not rows:
        return ""

    parts = [rows[0]["classification"]]
    previous = rows[0]["classification"]

    for row in rows[1:]:

        if row["classification"] != previous:

            alpha = row["alpha"]
            label = f"{alpha:.2f}" if alpha != "" else "?"
            parts.append(f"{row['classification']}@{label}")
            previous = row["classification"]

    return " -> ".join(parts)


# ======================================================================
# POSITIONS-SCANS
# ======================================================================

def build_positions(canvas_W, canvas_H, training_x, training_y):
    """
    Die Mitte muss an erster Stelle stehen (Referenz fuer alle anderen).
    Gibt nur die Positionen aus SCAN_POSITIONS zurueck.
    """

    positions = all_positions(canvas_W, canvas_H, training_x, training_y)

    if SCAN_POSITIONS == "all":
        return positions

    names = [name for name, _, _ in positions]
    unknown = set(SCAN_POSITIONS) - set(names)

    if unknown:
        raise ValueError(
            f"SCAN_POSITIONS enthaelt unbekannte Positionen: {sorted(unknown)}; "
            f"moeglich sind {names}"
        )

    if "center" not in SCAN_POSITIONS:
        raise ValueError(
            "SCAN_POSITIONS muss 'center' enthalten (Referenz fuer die "
            "Positionsvergleiche)."
        )

    return [p for p in positions if p[0] in SCAN_POSITIONS]


def all_positions(canvas_W, canvas_H, training_x, training_y):
    """Alle 9 Startpositionen, die Mitte zuerst."""

    return [
        ("center", training_x, training_y),
        ("top_left", 0, 0),
        ("top_center", training_x, 0),
        ("top_right", canvas_W - 1, 0),
        ("center_left", 0, training_y),
        ("center_right", canvas_W - 1, training_y),
        ("bottom_left", 0, canvas_H - 1),
        ("bottom_center", training_x, canvas_H - 1),
        ("bottom_right", canvas_W - 1, canvas_H - 1),
    ]


def build_scan_list(ctx):

    scans = []

    for a, b in itertools.combinations(ctx.attractors, 2):

        tag = f"{a.name.lower()}_to_{b.name.lower()}"

        scans.append(SimpleNamespace(
            name=f"normal_{tag}",
            scan_type="normal",
            from_attractor=a.name,
            to_attractor=b.name,
            start=a.seed_pixel,
            end=b.seed_pixel,
            pixel=None,
        ))

        scans.append(SimpleNamespace(
            name=f"complementary_{tag}",
            scan_type="complementary",
            from_attractor=a.name,
            to_attractor=b.name,
            start=complementary_pixel(a.seed_pixel),
            end=complementary_pixel(b.seed_pixel),
            pixel=None,
        ))

    scans.append(SimpleNamespace(
        name="black",
        scan_type="black",
        from_attractor="",
        to_attractor="",
        start=None,
        end=None,
        pixel=make_pixel(ctx.data_channels, 0.0, 0.0, 0.0, a=1.0),
    ))

    scans.append(SimpleNamespace(
        name="transparent",
        scan_type="transparent",
        from_attractor="",
        to_attractor="",
        start=None,
        end=None,
        pixel=make_pixel(ctx.data_channels, 0.0, 0.0, 0.0, a=0.0),
    ))

    return scans


def run_scan(ctx, scan):

    series = scan.pixel is None

    if series:
        alphas = list(ALPHAS)
        pixels = [
            ((1.0 - alpha) * scan.start + alpha * scan.end).astype(np.float32)
            for alpha in alphas
        ]
    else:
        alphas = [None]
        pixels = [scan.pixel]

    seeds = []
    meta = []
    needs_frames = []
    gif_positions = evolution_positions(ctx)

    # Mit Klassenfilter steht erst nach dem Lauf fest, wer ein GIF bekommt;
    # dann zeichnet der erste Durchgang keine Bilder auf (siehe unten).
    record_now = EVOLUTION_GIF_CLASSES is None

    for position_name, seed_x, seed_y in ctx.positions:
        for index, pixel in enumerate(pixels):
            seeds.append((pixel, seed_x, seed_y))
            meta.append((position_name, seed_x, seed_y, index))
            needs_frames.append(record_now and position_name in gif_positions)

    reference_states = {}
    reference_classes = {}
    rows_by_position = defaultdict(list)
    rows_by_seed = {}
    frames_by_position = defaultdict(list)

    started = time.time()

    print(f"\n--- {ctx.tag} / {scan.name} ({len(seeds)} runs) ---")

    for seed_index, raw_state, dynamics, extras in iterate_rollouts(
        ctx, seeds, needs_frames
    ):

        position_name, seed_x, seed_y, index = meta[seed_index]
        pixel = pixels[index]
        alpha = alphas[index]

        analysis = classify_state(ctx, raw_state, seed_x, seed_y)
        aligned_state = analysis.pop("aligned_state")
        classification = analysis["classification"]

        dynamics = reconcile_with_classification(
            dynamics, extras.labels, classification, ctx.class_order
        )

        evolution_gif = ""

        if extras.frames is not None:

            gif_dir = ctx.output_dir / position_name / scan.name / "evolution"

            if series:
                gif_path = (
                    gif_dir / f"alpha_{index:03d}_{classification.lower()}.gif"
                )
                header = f"alpha={alpha:.2f} | {position_name}"
            else:
                gif_path = gif_dir / "evolution.gif"
                header = f"{scan.name} | {position_name}"

            ctx.gif_writer.submit(
                gif_path,
                extras.frames,
                extras.labels,
                ctx.class_order,
                header,
                dict(dynamics),
                classification,
            )
            evolution_gif = str(gif_path.relative_to(ctx.output_dir))

        if position_name == "center":

            reference_states[index] = raw_state
            reference_classes[index] = classification
            comparison = dict(ZERO_COMPARISON)
            class_changed = False

        else:

            comparison = compare_to_center(
                raw_state,
                aligned_state,
                reference_states[index],
                ctx.data_channels,
            )
            class_changed = classification != reference_classes[index]

        output_dir = ctx.output_dir / position_name / scan.name

        if SAVE_IMAGES:

            raw_image = state_to_rgba(raw_state)
            aligned_image = state_to_rgba(aligned_state)

            if series:

                filename = f"alpha_{index:03d}_{classification.lower()}.png"
                raw_path = output_dir / "raw_full_matrix" / filename
                aligned_path = output_dir / "aligned_analysis_only" / filename

            else:

                raw_path = output_dir / "raw_full_matrix.png"
                aligned_path = output_dir / "aligned_analysis_only.png"

            raw_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_path.parent.mkdir(parents=True, exist_ok=True)

            save_rgba(raw_image, raw_path)
            save_rgba(aligned_image, aligned_path)

        if series and SAVE_GIFS:

            frame = Image.fromarray(state_to_rgba(raw_state)).convert("RGB")
            draw = ImageDraw.Draw(frame)
            # Die Frames sind nur so breit wie das Canvas (z.B. 140 px), alles
            # rechts davon wird abgeschnitten. Deshalb steht in jeder Zeile
            # das Wichtigste vorne: zuerst alpha, dann die Zahlen. Der
            # Attraktorname steht schon in Zeile 2 und wird in Zeile 3 nicht
            # wiederholt - bei langen Namen wie ALIEN_MONSTER wuerde sonst die
            # Schrittzahl abgeschnitten.
            draw.text(
                (5, 5),
                f"alpha={alpha:.2f} | {position_name} | {scan.name}",
            )
            draw.text((5, 20), classification)

            if dynamics.get("reaches_pure_attractor"):
                attractor_text = (
                    f"attractor at {dynamics['steps_to_pure_attractor']}"
                    f"/{STEPS}"
                )
            elif "steps_to_attractor" in dynamics:
                attractor_text = "no pure attractor"
            else:
                attractor_text = ""

            if attractor_text:
                draw.text((5, 35), attractor_text)

            # Aenderungsbasiertes Mass, bewusst anders beschriftet. Steht bei
            # STEPS = Training-t fast immer auf STEPS.
            draw.text(
                (5, 50),
                f"settled at {dynamics['steps_to_stable']}/{STEPS}",
            )
            frames_by_position[position_name].append(frame)

        row = {
            "scan": scan.name,
            "scan_type": scan.scan_type,
            "from_attractor": scan.from_attractor,
            "to_attractor": scan.to_attractor,
            "position": position_name,
            "seed_x": seed_x,
            "seed_y": seed_y,
            "alpha": float(alpha) if series else "",
            **pixel_columns(pixel, ctx.data_channels),
            "classification": classification,
            "class_changed_vs_center": class_changed,
            **analysis,
            **dynamics,
            **comparison,
            "evolution_gif": evolution_gif,
        }

        rows_by_position[position_name].append(row)
        rows_by_seed[seed_index] = row
        ctx.all_results.append(row)

        if index == len(pixels) - 1:

            rows = rows_by_position[position_name]
            changes = sum(r["class_changed_vs_center"] for r in rows)
            mean_aligned = np.mean(
                [r["aligned_visible_mse_vs_center"] for r in rows]
            )
            convergence = summarise_convergence(rows)

            pure_steps = convergence["mean_steps_to_pure_attractor"]
            pure_share = convergence["fraction_reaching_pure_attractor"]
            attractor_text = (
                f"pure attractor after {pure_steps:.1f}/{STEPS} "
                f"({pure_share:.0%} of runs reach one)"
                if isinstance(pure_steps, float)
                else "no run reaches a pure attractor"
            )

            print(
                f"{position_name:14s} | changes vs center={changes:3d} | "
                f"mean aligned MSE={mean_aligned:.6f} | "
                f"settled after {convergence['mean_steps_to_stable']:.1f}/"
                f"{STEPS} "
                f"({convergence['runs_not_settled']} not settled) | "
                f"{attractor_text} | "
                f"{describe_transitions(rows)}"
            )

    # Zweiter Durchgang: nur die Laeufe mit passender End-Klasse noch einmal
    # rechnen, diesmal mit Bildaufzeichnung. Vor dem Schreiben der CSVs, damit
    # die Spalte evolution_gif schon gefuellt ist.
    selected = []

    if EVOLUTION_GIF_CLASSES is not None:

        selected = [
            i for i, (position_name, _, _, _) in enumerate(meta)
            if position_name in gif_positions
            and gif_selected_by_class(rows_by_seed[i]["classification"])
        ]

        def describe(i):

            position_name, _, _, index = meta[i]
            classification = rows_by_seed[i]["classification"]
            gif_dir = ctx.output_dir / position_name / scan.name / "evolution"

            if series:
                return (
                    gif_dir / f"alpha_{index:03d}_{classification.lower()}.gif",
                    f"alpha={alphas[index]:.2f} | {position_name}",
                )

            return gif_dir / "evolution.gif", f"{scan.name} | {position_name}"

        mismatches = record_selected_gifs(
            ctx, seeds, selected, rows_by_seed, describe
        )

        print(
            f"evolution GIFs for {', '.join(EVOLUTION_GIF_CLASSES)}: "
            f"{len(selected)} of {len(seeds)} runs"
            + (f" | !!! {mismatches} replay mismatches" if mismatches else "")
        )

    for position_name, _, _ in ctx.positions:

        output_dir = ctx.output_dir / position_name / scan.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if series:

            write_csv(output_dir / "results.csv", rows_by_position[position_name])

            frames = frames_by_position.get(position_name)

            if frames:
                frames[0].save(
                    output_dir / "raw_full_matrix_scan.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0,
                )

    written, failures = ctx.gif_writer.finish()
    ctx.gif_failures.extend(failures)

    print(
        f"({time.time() - started:.1f} s, "
        f"{written} evolution GIFs written, "
        f"{len(failures)} failed)"
    )


def build_position_summary(ctx, scans):

    grouped = defaultdict(list)

    for row in ctx.all_results:
        grouped[(row["position"], row["scan"])].append(row)

    summary_rows = []

    for position_name, seed_x, seed_y in ctx.positions:

        for scan in scans:

            selected = grouped.get((position_name, scan.name))

            if not selected:
                continue

            aligned_values = [r["aligned_visible_mse_vs_center"] for r in selected]
            raw_values = [r["raw_visible_mse_vs_center"] for r in selected]

            summary_rows.append({
                "position": position_name,
                "seed_x": seed_x,
                "seed_y": seed_y,
                "scan": scan.name,
                "scan_type": scan.scan_type,
                "from_attractor": scan.from_attractor,
                "to_attractor": scan.to_attractor,
                "runs": len(selected),
                "classification_changes_vs_center": sum(
                    bool(r["class_changed_vs_center"]) for r in selected
                ),
                "mean_raw_visible_mse_vs_center": float(np.mean(raw_values)),
                "mean_aligned_visible_mse_vs_center": float(np.mean(aligned_values)),
                "max_aligned_visible_mse_vs_center": float(np.max(aligned_values)),
                **{
                    key: value
                    for key, value in summarise_convergence(selected).items()
                    if key != "runs"
                },
                **class_count_columns(
                    ctx, [r["classification"] for r in selected]
                ),
            })

    return summary_rows


# ======================================================================
# RGB-SPEKTRUM-SCAN
# ======================================================================

def save_legend(path, class_colours):

    entries = list(class_colours.items())
    image = Image.new("RGB", (280, 24 * len(entries) + 8), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for i, (name, colour) in enumerate(entries):

        y = 4 + i * 24
        draw.rectangle([6, y, 26, y + 18], fill=colour[:3], outline=(0, 0, 0))
        draw.text((34, y + 4), name, fill=(0, 0, 0))

    image.save(path)


def run_rgb_spectrum_scan(ctx, position_name):

    if SPECTRUM_LEVELS ** 3 != SPECTRUM_GRID_SIZE ** 2:
        raise ValueError("SPECTRUM_LEVELS^3 must equal SPECTRUM_GRID_SIZE^2")

    if SPECTRUM_TILE_SIZE != SPECTRUM_LEVELS:
        raise ValueError("SPECTRUM_TILE_SIZE must equal SPECTRUM_LEVELS")

    tiles_per_axis = SPECTRUM_GRID_SIZE // SPECTRUM_TILE_SIZE

    if tiles_per_axis ** 2 != SPECTRUM_LEVELS:
        raise ValueError("The tile layout must contain one tile per blue level")

    if ctx.data_channels < 3:
        raise RuntimeError("RGB spectrum scan requires at least 3 data channels")

    positions = {name: (x, y) for name, x, y in ctx.positions}

    if position_name not in positions:
        raise ValueError(f"Unknown spectrum position {position_name!r}")

    seed_x, seed_y = positions[position_name]

    folder = (
        "rgb_spectrum_scan" if position_name == "center"
        else f"rgb_spectrum_scan_{position_name}"
    )
    output_dir = ctx.output_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    size = SPECTRUM_GRID_SIZE
    input_matrix = np.zeros((size, size, 4), dtype=np.uint8)
    classification_matrix = np.zeros((size, size, 4), dtype=np.uint8)
    steps_matrix = np.zeros((size, size), dtype=np.float64)
    attractor_matrix = np.zeros((size, size), dtype=np.float64)

    seeds = []
    cells = []

    for y in range(size):
        for x in range(size):

            tile_col = x // SPECTRUM_TILE_SIZE
            tile_row = y // SPECTRUM_TILE_SIZE
            blue_index = tile_row * tiles_per_axis + tile_col
            red_index = x % SPECTRUM_LEVELS
            green_index = y % SPECTRUM_LEVELS

            r = red_index / (SPECTRUM_LEVELS - 1)
            g = green_index / (SPECTRUM_LEVELS - 1)
            b = blue_index / (SPECTRUM_LEVELS - 1)

            pixel = make_pixel(ctx.data_channels, r, g, b, a=1.0)

            seeds.append((pixel, seed_x, seed_y))
            cells.append(
                (x, y, tile_row, tile_col, red_index, green_index, blue_index, r, g, b)
            )

    print(f"\n--- {ctx.tag} / RGB spectrum scan at {position_name} "
          f"({len(seeds)} colours, final images: {RGB_SAVE_FINAL_IMAGES}, "
          f"evolution GIFs: {RGB_SAVE_EVOLUTION_GIFS}) ---")

    rows = []
    started = time.time()
    # Mit Klassenfilter: erst rechnen, dann nur die passenden Farben erneut
    # mit Bildaufzeichnung (siehe unten).
    needs_frames = [
        RGB_SAVE_EVOLUTION_GIFS and EVOLUTION_GIF_CLASSES is None
    ] * len(seeds)

    final_dir = output_dir / "final_images"
    gif_dir = output_dir / "evolution"

    if RGB_SAVE_FINAL_IMAGES:
        final_dir.mkdir(parents=True, exist_ok=True)

    for seed_index, raw_state, dynamics, extras in iterate_rollouts(
        ctx, seeds, needs_frames
    ):

        (x, y, tile_row, tile_col, red_index, green_index,
         blue_index, r, g, b) = cells[seed_index]

        analysis = classify_state(ctx, raw_state, seed_x, seed_y)
        analysis.pop("aligned_state")
        classification = analysis["classification"]

        dynamics = reconcile_with_classification(
            dynamics, extras.labels, classification, ctx.class_order
        )

        # Dateiname aus den drei Farbstufen (0 .. SPECTRUM_LEVELS - 1), damit
        # jede der 4096 Farben eindeutig und sortierbar ist.
        stem = (
            f"r{red_index:02d}_g{green_index:02d}_b{blue_index:02d}_"
            f"{classification.lower()}"
        )

        final_image = ""

        if RGB_SAVE_FINAL_IMAGES:
            final_path = final_dir / f"{stem}.png"
            save_rgba(state_to_rgba(raw_state), final_path)
            final_image = str(final_path.relative_to(ctx.output_dir))

        evolution_gif = ""

        if extras.frames is not None:
            gif_path = gif_dir / f"{stem}.gif"
            ctx.gif_writer.submit(
                gif_path,
                extras.frames,
                extras.labels,
                ctx.class_order,
                f"R{red_index} G{green_index} B{blue_index}",
                dict(dynamics),
                classification,
            )
            evolution_gif = str(gif_path.relative_to(ctx.output_dir))

        input_matrix[y, x] = (
            int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255
        )
        classification_matrix[y, x] = ctx.class_colours.get(
            classification, UNKNOWN_COLOUR
        )
        steps_matrix[y, x] = dynamics["steps_to_stable"]

        # Farben, die nie auf einem echten Attraktor landen, bekommen den
        # Maximalwert und erscheinen damit weiss.
        attractor_matrix[y, x] = (
            dynamics["steps_to_pure_attractor"]
            if dynamics.get("reaches_pure_attractor") else STEPS
        )

        rows.append({
            "x": x,
            "y": y,
            "position": position_name,
            "seed_x": seed_x,
            "seed_y": seed_y,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "red_index": red_index,
            "green_index": green_index,
            "blue_index": blue_index,
            "R": float(r),
            "G": float(g),
            "B": float(b),
            "A": 1.0 if ctx.data_channels >= 4 else "",
            **analysis,
            **dynamics,
            "final_image": final_image,
            "evolution_gif": evolution_gif,
        })

        done = seed_index + 1

        if done % 512 == 0 or done == len(seeds):
            print(
                f"RGB scan progress: {done}/{len(seeds)} "
                f"({time.time() - started:.0f} s)"
            )

    if RGB_SAVE_EVOLUTION_GIFS and EVOLUTION_GIF_CLASSES is not None:

        # rows liegen in derselben Reihenfolge wie seeds.
        selected = [
            i for i, row in enumerate(rows)
            if gif_selected_by_class(row["classification"])
        ]

        def describe(i):

            (_, _, _, _, red_index, green_index, blue_index,
             _, _, _) = cells[i]
            stem = (
                f"r{red_index:02d}_g{green_index:02d}_b{blue_index:02d}_"
                f"{rows[i]['classification'].lower()}"
            )
            return (
                gif_dir / f"{stem}.gif",
                f"R{red_index} G{green_index} B{blue_index}",
            )

        mismatches = record_selected_gifs(
            ctx, seeds, selected, dict(enumerate(rows)), describe
        )

        print(
            f"RGB evolution GIFs for {', '.join(EVOLUTION_GIF_CLASSES)}: "
            f"{len(selected)} of {len(seeds)} colours"
            + (f" | !!! {mismatches} replay mismatches" if mismatches else "")
        )

    written, failures = ctx.gif_writer.finish()
    ctx.gif_failures.extend(failures)

    if RGB_SAVE_EVOLUTION_GIFS:
        print(f"RGB evolution GIFs written: {written}, failed: {len(failures)}")

    csv_path = output_dir / "rgb_spectrum_results.csv"
    write_csv(csv_path, rows)       # nach dem zweiten Durchgang: GIF-Pfade sind drin

    paths = {
        "input_64_path": output_dir / "rgb_spectrum_input_64x64.png",
        "class_64_path": output_dir / "rgb_spectrum_classification_64x64.png",
        "input_large_path": output_dir / "rgb_spectrum_input_large.png",
        "class_large_path": output_dir / "rgb_spectrum_classification_large.png",
        "legend_path": output_dir / "rgb_spectrum_legend.png",
        "steps_64_path": output_dir / "rgb_spectrum_steps_to_stable_64x64.png",
        "steps_large_path": output_dir / "rgb_spectrum_steps_to_stable_large.png",
        "attractor_64_path":
            output_dir / "rgb_spectrum_steps_to_pure_attractor_64x64.png",
        "attractor_large_path":
            output_dir / "rgb_spectrum_steps_to_pure_attractor_large.png",
    }

    large = (size * SPECTRUM_UPSCALE, size * SPECTRUM_UPSCALE)

    Image.fromarray(input_matrix).save(paths["input_64_path"])
    Image.fromarray(classification_matrix).save(paths["class_64_path"])

    Image.fromarray(input_matrix).resize(
        large, resample=Image.Resampling.NEAREST
    ).save(paths["input_large_path"])

    Image.fromarray(classification_matrix).resize(
        large, resample=Image.Resampling.NEAREST
    ).save(paths["class_large_path"])

    save_legend(paths["legend_path"], ctx.class_colours)

    # Karte der Konvergenzzeit: schwarz = sofort stabil,
    # weiss = erst nach STEPS Schritten (oder gar nicht) stabil.
    def save_steps_map(matrix, small_path, large_path):

        image = np.round(
            np.clip(matrix / max(STEPS, 1), 0.0, 1.0) * 255
        ).astype(np.uint8)

        Image.fromarray(image, mode="L").save(small_path)
        Image.fromarray(image, mode="L").resize(
            large, resample=Image.Resampling.NEAREST
        ).save(large_path)

    save_steps_map(
        steps_matrix, paths["steps_64_path"], paths["steps_large_path"]
    )
    save_steps_map(
        attractor_matrix,
        paths["attractor_64_path"],
        paths["attractor_large_path"],
    )

    counts = class_count_columns(ctx, [row["classification"] for row in rows])
    convergence = summarise_convergence(rows)

    summary = {
        "position": position_name,
        "seed_x": seed_x,
        "seed_y": seed_y,
        "total_runs": len(rows),
        **counts,
        **{
            key: value
            for key, value in convergence.items()
            if key != "runs"
        },
        "csv_path": str(csv_path.resolve()),
        **{key: str(path.resolve()) for key, path in paths.items()},
    }

    summary_path = output_dir / "rgb_spectrum_summary.txt"

    with open(summary_path, "w") as file:

        file.write("======================================\n")
        file.write("RGB SPECTRUM SCAN SUMMARY\n")
        file.write("======================================\n\n")
        file.write(f"Model: {ctx.tag}\n")
        file.write(f"Bundle: {ctx.bundle}\n")
        file.write(f"Padding: {ctx.padding}\n")
        file.write(f"Position: {position_name}\n")
        file.write(f"Seed position: ({seed_x}, {seed_y})\n")
        file.write(f"Total tested colours: {len(rows)}\n")
        file.write(f"RGB levels per channel: {SPECTRUM_LEVELS}\n")
        file.write("Alpha: 1.0 for RGBA data\n\n")

        file.write("CLASSIFICATION COLOURS\n")
        file.write("--------------------------------------\n")
        for name, colour in ctx.class_colours.items():
            file.write(f"{name}: {colour[:3]}\n")

        file.write("\nCLASSIFICATION COUNTS\n")
        file.write("--------------------------------------\n")
        for name in ctx.class_colours:
            file.write(f"{name}: {counts[f'{name.lower()}_count']}\n")

        file.write("\nTIME TO STABLE PATTERN\n")
        file.write("--------------------------------------\n")
        file.write(f"Inference steps per run: {STEPS}\n")
        for key, value in convergence.items():
            if key != "runs":
                file.write(f"{key}: {value}\n")
        file.write(
            "Maps: rgb_spectrum_steps_to_stable_*.png and "
            "rgb_spectrum_steps_to_pure_attractor_*.png "
            "(black = immediately, white = only at the very end or, in the "
            "pure-attractor map, never)\n"
        )

        file.write("\nMATRIX LAYOUT\n")
        file.write("--------------------------------------\n")
        file.write(
            f"The {size}x{size} matrix is divided into a "
            f"{tiles_per_axis}x{tiles_per_axis} grid of "
            f"{SPECTRUM_TILE_SIZE}x{SPECTRUM_TILE_SIZE} tiles.\n"
        )
        file.write(f"Each tile represents one of the {SPECTRUM_LEVELS} blue levels.\n")
        file.write("Inside every tile, x represents red and y represents green.\n\n")

        file.write("SAVED FILES\n")
        file.write("--------------------------------------\n")
        file.write(f"CSV: {csv_path.resolve()}\n")
        for key, path in paths.items():
            file.write(f"{key}: {path.resolve()}\n")

    summary["summary_path"] = str(summary_path.resolve())

    print("Classification image:", paths["class_large_path"].resolve())

    return rows, summary


# ======================================================================
# PRO MODELL
# ======================================================================

def prepare_model(plan, emoji_dir, step_keys):

    cfg = plan.cfg
    data = read_data_settings(cfg)
    attractors = load_attractors(plan.specs, data, emoji_dir)

    channels, H, W = attractors[0].target.shape

    if data.data_channels > channels:
        raise RuntimeError(
            f"data_channels={data.data_channels} but emojis have {channels} channels"
        )

    pad_top, pad_bottom, pad_left, pad_right = data.pad
    canvas_H = H + pad_top + pad_bottom
    canvas_W = W + pad_left + pad_right

    warnings = []

    seed_positions = {a.seed_yx for a in attractors}

    if len(seed_positions) > 1:
        warnings.append(
            f"Attractors have different seed positions {sorted(seed_positions)}; "
            f"using {attractors[0].seed_yx} of {attractors[0].name}."
        )

    seed_y_img, seed_x_img = attractors[0].seed_yx
    training_x = seed_x_img + pad_left
    training_y = seed_y_img + pad_top

    padding_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    targets_full = np.stack(
        [np.pad(a.target, padding_width, mode="constant") for a in attractors]
    ).astype(np.float32)

    class_colours = {}

    for i, attractor in enumerate(attractors):

        attractor.colour = (
            ATTRACTOR_COLOURS[i] if i < len(ATTRACTOR_COLOURS) else UNKNOWN_COLOUR
        )
        class_colours[attractor.name] = attractor.colour

    class_colours[INTERMEDIATE] = INTERMEDIATE_COLOUR
    class_colours[NO_PATTERN] = NO_PATTERN_COLOUR

    for attractor in attractors:

        if data.data_channels >= 4 and attractor.seed_pixel[3] <= ACTIVE_ALPHA_THRESHOLD:
            warnings.append(
                f"Seed pixel of {attractor.name} is (almost) transparent: "
                f"{pixel_to_255(attractor.seed_pixel)}"
            )

        if not (attractor.initial_mode == "pixel" or
                attractor.initial_description == "patch 1x1"):
            warnings.append(
                f"{attractor.name} was trained with initial condition "
                f"'{attractor.initial_description}'; this scan seeds only its "
                f"centre pixel."
            )

    for a, b in itertools.combinations(attractors, 2):

        if np.allclose(a.seed_pixel, b.seed_pixel, atol=0.5 / 255):
            warnings.append(
                f"{a.name} and {b.name} have the same seed pixel "
                f"{pixel_to_255(a.seed_pixel)}; a single pixel cannot "
                f"distinguish them."
            )

    padding = plan.padding

    if padding not in {"REPLICATE", "CIRCULAR"}:
        warnings.append(
            f"Padding {padding!r} is neither REPLICATE nor CIRCULAR; "
            f"analysis alignment uses no wrap-around."
        )

    model = load_model(cfg, plan.model_path)

    output_dir = OUTPUT_ROOT / plan.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = SimpleNamespace(
        tag=plan.tag,
        bundle=plan.bundle,
        cfg=cfg,
        data=data,
        family=plan.family,
        padding=padding,
        circular=padding == "CIRCULAR",
        attractors=attractors,
        names=[a.name for a in attractors],
        # Reihenfolge der Klassen-Indizes aus classify_state_jax:
        # erst die Attraktoren, dann INTERMEDIATE, dann NO_VISIBLE_PATTERN.
        class_order=[a.name for a in attractors] + [INTERMEDIATE, NO_PATTERN],
        targets_full=targets_full,
        class_colours=class_colours,
        model=model,
        n_channels=int(cfg.model.channels),
        data_channels=data.data_channels,
        image_H=H,
        image_W=W,
        canvas_H=canvas_H,
        canvas_W=canvas_W,
        training_x=training_x,
        training_y=training_y,
        positions=build_positions(canvas_W, canvas_H, training_x, training_y),
        step_keys=step_keys,
        output_dir=output_dir,
        warnings=warnings,
        all_results=[],
        gif_writer=GifWriter(GIF_WRITER_THREADS),
        gif_failures=[],
        replay_mismatches=0,
    )

    print_model_info(ctx)
    write_model_info(ctx)

    return ctx


def print_model_info(ctx):

    print("\n======================================")
    print("CONFIG")
    print("======================================")
    print("Bundle:", ctx.bundle)
    print("Output:", ctx.output_dir.resolve())
    print("Family:", ctx.family)
    print("Padding:", ctx.padding)
    print("Channels:", ctx.n_channels)
    print("Fire rate:", cfg_value(ctx.cfg, "model.fire_rate", default="?"))
    print("Downsample:", ctx.data.downsample)
    print("Iterations:", ctx.data.iterations)
    print("Training t:", ctx.data.run_t)

    print("\n======================================")
    print("ATTRACTORS / SEED PIXELS")
    print("======================================")

    for a in ctx.attractors:
        print(
            f"{a.name:12s} | target={a.target_file:16s} | "
            f"initial={a.initial_file} ({a.initial_description}) | "
            f"seed (y,x)={a.seed_yx} | RGBA={pixel_to_255(a.seed_pixel)}"
        )

    print("\n======================================")
    print("CANVAS")
    print("======================================")
    print("Full matrix:", ctx.canvas_H, "x", ctx.canvas_W)
    print("Training seed position:", (ctx.training_x, ctx.training_y))

    for name, x, y in ctx.positions:
        print(f"  {name} -> {(x, y)}")

    if ctx.warnings:
        print("\nWARNINGS:")
        for warning in ctx.warnings:
            print("  !", warning)


def write_model_info(ctx):

    rows = []

    for a in ctx.attractors:
        rows.append({
            "index": a.index,
            "name": a.name,
            "target_file": a.target_file,
            "initial_file": a.initial_file,
            "initial_condition": a.initial_description,
            "seed_y_in_image": a.seed_yx[0],
            "seed_x_in_image": a.seed_yx[1],
            "seed_x_on_canvas": ctx.training_x,
            "seed_y_on_canvas": ctx.training_y,
            **{
                f"seed_{channel}": float(value)
                for channel, value in zip("RGBA", a.seed_pixel)
            },
            "seed_rgba_255": pixel_to_255(a.seed_pixel),
            "classification_colour": a.colour[:3],
        })

    write_csv(ctx.output_dir / "attractors.csv", rows)

    with open(ctx.output_dir / "model_info.txt", "w") as file:

        file.write(f"Model tag: {ctx.tag}\n")
        file.write(f"Bundle: {Path(ctx.bundle).resolve()}\n")
        file.write(f"Family: {ctx.family}\n")
        file.write(f"Padding: {ctx.padding}\n")
        file.write(f"Attractors: {', '.join(ctx.names)}\n")
        file.write(f"Canvas: {ctx.canvas_H} x {ctx.canvas_W}\n")
        file.write(f"Training seed position: ({ctx.training_x}, {ctx.training_y})\n")

        if ctx.warnings:
            file.write("\nWarnings:\n")
            for warning in ctx.warnings:
                file.write(f"  - {warning}\n")


def build_convergence_rows(ctx, scans, spectrum_results):
    """Konvergenz je Position und Scan, plus je Spektrum-Scan."""

    grouped = defaultdict(list)

    for row in ctx.all_results:
        grouped[(row["position"], row["scan"])].append(row)

    rows = []

    for position_name, seed_x, seed_y in ctx.positions:

        for scan in scans:

            selected = grouped.get((position_name, scan.name))

            if not selected:
                continue

            rows.append({
                "position": position_name,
                "seed_x": seed_x,
                "seed_y": seed_y,
                "scan": scan.name,
                "scan_type": scan.scan_type,
                **summarise_convergence(selected),
            })

    for spectrum_rows, summary in spectrum_results:

        rows.append({
            "position": summary["position"],
            "seed_x": summary["seed_x"],
            "seed_y": summary["seed_y"],
            "scan": "rgb_spectrum",
            "scan_type": "rgb_spectrum",
            **summarise_convergence(spectrum_rows),
        })

    return rows


def write_report(ctx, summary_rows, spectrum_results, convergence_rows):

    report_path = ctx.output_dir / "report.txt"

    with open(report_path, "w") as f:

        f.write("======================================\n")
        f.write("SPATIAL POSITION SCAN REPORT\n")
        f.write("======================================\n\n")

        f.write("CONFIGURATION\n")
        f.write("--------------------------------------\n")
        f.write(f"Model tag: {ctx.tag}\n")
        f.write(f"Bundle: {Path(ctx.bundle).resolve()}\n")
        f.write(f"Model family: {ctx.family}\n")
        f.write(f"Padding: {ctx.padding}\n")
        f.write(f"Channels: {ctx.n_channels}\n")
        f.write(f"Fire rate: {cfg_value(ctx.cfg, 'model.fire_rate', default='?')}\n")
        f.write(f"Downsample: {ctx.data.downsample}\n")
        f.write(f"Training iterations: {ctx.data.iterations}\n")
        f.write(f"Training t: {ctx.data.run_t}\n")
        f.write(f"Inference steps: {STEPS}\n")
        f.write(f"Canvas size: {ctx.canvas_H} x {ctx.canvas_W}\n")
        f.write(f"Training seed position: ({ctx.training_x}, {ctx.training_y})\n")
        f.write(
            f"Convergence tolerances: abs={CONVERGENCE_ABS_TOL}, "
            f"rel={CONVERGENCE_REL_TOL} x residual change "
            f"(median of last {CONVERGENCE_TAIL_STEPS} steps), "
            f"channels={'visible' if CONVERGENCE_USE_VISIBLE_CHANNELS else 'all'}\n"
        )
        f.write(
            f"Exact distance to final pattern measured: "
            f"{MEASURE_DISTANCE_TO_FINAL}\n\n"
        )

        f.write("ATTRACTORS / SEED PIXELS\n")
        f.write("--------------------------------------\n")
        for a in ctx.attractors:
            f.write(
                f"{a.name}: target={a.target_file}, initial={a.initial_file} "
                f"({a.initial_description}), seed={np.asarray(a.seed_pixel)}, "
                f"complement={complementary_pixel(a.seed_pixel)}\n"
            )

        if ctx.warnings:
            f.write("\nWARNINGS\n")
            f.write("--------------------------------------\n")
            for warning in ctx.warnings:
                f.write(f"{warning}\n")

        f.write("\nPOSITIONS\n")
        f.write("--------------------------------------\n")
        for name, x, y in ctx.positions:
            f.write(f"{name}: ({x}, {y})\n")

        f.write("\nTIME TO STABLE PATTERN\n")
        f.write("--------------------------------------\n")
        f.write(
            "steps_to_stable = last step after which the state stopped "
            f"changing (out of {STEPS} inference steps).\n"
        )
        f.write(
            "steps_to_attractor = last step at which the classification "
            "still differed from the final one. Robust against fire_rate "
            "jitter, since a class label does not flicker.\n"
        )
        f.write(
            "steps_to_pure_attractor = same, but only for runs that end on "
            "a real attractor: the last step at which the state was not yet "
            "assigned to that pattern, i.e. was INTERMEDIATE, "
            "NO_VISIBLE_PATTERN or a different attractor. Runs that end as "
            "a mixture or an empty canvas have no such step and are left "
            "blank instead of contributing a number. Always read the mean "
            "together with fraction_reaching_pure_attractor - it only "
            "covers the runs that get there.\n"
        )
        f.write(
            "A run can lock onto its attractor long before the state stops "
            "changing numerically; if steps_to_stable equals the step count "
            "for every run, the inference horizon is most likely too short "
            "(compare STEPS with the training t above).\n\n"
        )

        f.write("Position scans, all runs together:\n")
        for key, value in summarise_convergence(ctx.all_results).items():
            f.write(f"  {key}: {value}\n")

        for rows, summary in spectrum_results:
            f.write(
                f"\nRGB spectrum scan at {summary['position']}:\n"
            )
            for key, value in summarise_convergence(rows).items():
                f.write(f"  {key}: {value}\n")

        f.write("\nPer scan and position:\n")
        for row in convergence_rows:
            f.write(
                f"  {row['position']:14s} | {row['scan']:34s} | "
                f"settled mean={row['mean_steps_to_stable']} | "
                f"not settled={row['runs_not_settled']} | "
                f"pure attractor mean={row['mean_steps_to_pure_attractor']} | "
                f"reached by={row['runs_reaching_pure_attractor']}"
                f"/{row['runs']}\n"
            )

        if WRITE_FULL_REPORT:

            f.write("\n\nALL RESULTS\n")
            f.write("======================================\n")
            for row in ctx.all_results:
                f.write("\n")
                for key, value in row.items():
                    f.write(f"{key}: {value}\n")

        f.write("\n\nPOSITION SUMMARY\n")
        f.write("======================================\n")
        for row in summary_rows:
            f.write("\n")
            for key, value in row.items():
                f.write(f"{key}: {value}\n")

        for rows, summary in spectrum_results:

            f.write("\n\nRGB SPECTRUM SCAN SUMMARY\n")
            f.write("======================================\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

            if WRITE_FULL_REPORT:
                f.write("\n\nRGB SPECTRUM ALL RESULTS\n")
                f.write("======================================\n")
                for row in rows:
                    f.write("\n")
                    for key, value in row.items():
                        f.write(f"{key}: {value}\n")

    return report_path


def run_all_tests(ctx):

    scans = build_scan_list(ctx)

    print("\nScans:", ", ".join(scan.name for scan in scans))

    for scan in scans:
        run_scan(ctx, scan)

    write_csv(ctx.output_dir / "all_position_results.csv", ctx.all_results)

    summary_rows = build_position_summary(ctx, scans)
    write_csv(ctx.output_dir / "position_summary.csv", summary_rows)

    print("\n======================================")
    print(f"POSITION EFFECT SUMMARY ({ctx.tag})")
    print("======================================")

    for row in summary_rows:

        if row["position"] == "center":
            continue

        print(
            f"{row['position']:14s} | {row['scan']:34s} | "
            f"class changes={row['classification_changes_vs_center']:3d} | "
            f"mean raw MSE={row['mean_raw_visible_mse_vs_center']:.6f} | "
            f"mean aligned MSE={row['mean_aligned_visible_mse_vs_center']:.6f}"
        )

    spectrum_results = []

    if RUN_RGB_SPECTRUM_SCAN:
        for position_name in SPECTRUM_POSITIONS:
            spectrum_results.append(run_rgb_spectrum_scan(ctx, position_name))

    convergence_rows = build_convergence_rows(ctx, scans, spectrum_results)
    write_csv(ctx.output_dir / "convergence_summary.csv", convergence_rows)

    print("\n======================================")
    print(f"TIME TO STABLE PATTERN ({ctx.tag})")
    print("======================================")
    print(f"(out of {STEPS} inference steps per run)")

    for row in convergence_rows:

        pure_steps = row["mean_steps_to_pure_attractor"]
        pure_share = row["fraction_reaching_pure_attractor"]
        attractor_text = (
            f"pure attractor={pure_steps:6.1f} | "
            f"reached by={pure_share:5.1%}"
            if isinstance(pure_steps, float)
            else "pure attractor=     - | reached by= 0.0%"
        )

        print(
            f"{row['position']:14s} | {row['scan']:34s} | "
            f"settled={row['mean_steps_to_stable']:6.1f} | "
            f"not settled={row['runs_not_settled']:4d} | "
            f"{attractor_text}"
        )

    report_path = write_report(
        ctx, summary_rows, spectrum_results, convergence_rows
    )

    print("\nModel output:", ctx.output_dir.resolve())
    print("Convergence summary:",
          (ctx.output_dir / "convergence_summary.csv").resolve())
    print("Text report:", report_path.resolve())

    return (
        summary_rows,
        [summary for _, summary in spectrum_results],
        convergence_rows,
        summarise_convergence(
            ctx.all_results
            + [row for rows, _ in spectrum_results for row in rows]
        ),
    )


# ======================================================================
# HAUPTPROGRAMM
# ======================================================================

def model_columns(plan):

    return {
        "model": plan.tag,
        "n_attractors": plan.n_attractors,
        "padding": plan.padding,
        "family": plan.family,
    }


def write_cross_model_outputs(overview, all_summaries, all_spectrum,
                              all_convergence):

    write_csv(OUTPUT_ROOT / "models_overview.csv", overview)
    write_csv(OUTPUT_ROOT / "all_models_position_summary.csv", all_summaries)
    write_csv(OUTPUT_ROOT / "all_models_rgb_spectrum_summary.csv", all_spectrum)
    write_csv(OUTPUT_ROOT / "all_models_convergence.csv", all_convergence)


def frames_per_evolution_gif():

    indices = list(range(0, STEPS + 1, max(EVOLUTION_FRAME_STRIDE, 1)))

    if indices[-1] != STEPS:
        indices.append(STEPS)

    return len(indices)


def estimate_evolution_output(plans):
    """
    Grobe Vorabschaetzung: wie viele GIFs / PNGs entstehen und wie viel
    Platz brauchen sie. Positionen: laut SCAN_POSITIONS.
    """

    scanned = 9 if SCAN_POSITIONS == "all" else len(SCAN_POSITIONS)
    positions_with_gifs = (
        scanned if EVOLUTION_GIF_POSITIONS == "all"
        else len(set(EVOLUTION_GIF_POSITIONS) & (
            set(SCAN_POSITIONS) if SCAN_POSITIONS != "all"
            else set(EVOLUTION_GIF_POSITIONS)
        ))
    )
    spectrum_scans = len(SPECTRUM_POSITIONS) if RUN_RGB_SPECTRUM_SCAN else 0
    colours = SPECTRUM_GRID_SIZE ** 2

    position_gifs = 0

    for plan in plans:
        pairs = plan.n_attractors * (plan.n_attractors - 1) // 2
        runs_per_position = pairs * 2 * len(ALPHAS) + 2   # + black, transparent
        position_gifs += runs_per_position * positions_with_gifs

    rgb_gifs = (
        len(plans) * spectrum_scans * colours if RGB_SAVE_EVOLUTION_GIFS else 0
    )

    if EVOLUTION_GIF_CLASSES is not None:
        # Wie viele Laeufe ein GIF bekommen, steht erst nach dem Rechnen
        # fest - hier mit dem erwarteten Anteil geschaetzt.
        position_gifs = int(round(position_gifs * EVOLUTION_EXPECTED_GIF_FRACTION))
        rgb_gifs = int(round(rgb_gifs * EVOLUTION_EXPECTED_GIF_FRACTION))
    rgb_pngs = (
        len(plans) * spectrum_scans * colours if RGB_SAVE_FINAL_IMAGES else 0
    )

    gif_bytes = (
        (position_gifs + rgb_gifs)
        * frames_per_evolution_gif()
        * EVOLUTION_BYTES_PER_FRAME_ESTIMATE
        * max(EVOLUTION_UPSCALE, 1) ** 2
    )
    png_bytes = rgb_pngs * 40_000

    return SimpleNamespace(
        position_gifs=position_gifs,
        rgb_gifs=rgb_gifs,
        rgb_pngs=rgb_pngs,
        total_bytes=gif_bytes + png_bytes,
    )


def check_disk_space(estimate):

    free = shutil.disk_usage(OUTPUT_ROOT).free

    print("\n======================================")
    print("EVOLUTION OUTPUT (estimate)")
    print("======================================")
    if EVOLUTION_GIF_CLASSES is not None:
        print(
            f"GIFs only for final classes {EVOLUTION_GIF_CLASSES}; counts "
            f"below assume {EVOLUTION_EXPECTED_GIF_FRACTION:.0%} of runs"
        )

    print(f"Evolution GIFs (positions): {estimate.position_gifs}")
    print(f"Evolution GIFs (RGB):       {estimate.rgb_gifs}")
    print(f"RGB final images (PNG):     {estimate.rgb_pngs}")
    print(f"Frames per GIF:             {frames_per_evolution_gif()}")
    print(f"Estimated size:             {estimate.total_bytes / 1e9:.1f} GB")
    print(f"Free on disk:               {free / 1e9:.1f} GB")

    if estimate.total_bytes * 1.2 > free and not IGNORE_DISK_SPACE_CHECK:
        raise RuntimeError(
            f"Not enough disk space: the evolution output needs roughly "
            f"{estimate.total_bytes / 1e9:.0f} GB (+20% margin), but only "
            f"{free / 1e9:.0f} GB are free under {OUTPUT_ROOT.resolve()}. "
            f"Reduce EVOLUTION_GIF_POSITIONS / RGB_SAVE_EVOLUTION_GIFS / "
            f"EVOLUTION_FRAME_STRIDE, free up space, or set "
            f"IGNORE_DISK_SPACE_CHECK = True."
        )


def main():

    if "DATA_PATH_BASE" not in os.environ:
        raise RuntimeError("DATA_PATH_BASE ist nicht gesetzt.")

    emoji_dir = Path(os.environ["DATA_PATH_BASE"]) / "Emojis"

    print(f"FORCE_CPU: {FORCE_CPU}")
    print(f"JAX devices: {jax.devices()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    plans = plan_models(emoji_dir)

    print("\n======================================")
    print(f"PLAN: {len(plans)} MODELS")
    print("======================================")

    for i, plan in enumerate(plans, 1):
        print(
            f"{i}. {plan.tag:28s} | {', '.join(s.name for s in plan.specs)} | "
            f"{plan.bundle}"
        )

    # Vor dem ersten Rollout: reicht der Platz fuer GIFs und PNGs?
    check_disk_space(estimate_evolution_output(plans))

    step_keys = make_step_keys(STEPS, SEED)

    overview = []
    all_summaries = []
    all_spectrum = []
    all_convergence = []
    errors_path = OUTPUT_ROOT / "errors.txt"

    if errors_path.exists():
        errors_path.unlink()

    total_started = time.time()

    for i, plan in enumerate(plans, 1):

        print("\n\n##########################################################")
        print(f"MODEL {i}/{len(plans)}: {plan.tag}")
        print("##########################################################")

        started = time.time()
        status = "ok"
        error = ""
        ctx = None
        model_convergence = empty_convergence_summary()

        try:

            ctx = prepare_model(plan, emoji_dir, step_keys)
            (summary_rows, spectrum_summaries, convergence_rows,
             model_convergence) = run_all_tests(ctx)

            for row in summary_rows:
                all_summaries.append({**model_columns(plan), **row})

            for summary in spectrum_summaries:
                all_spectrum.append({**model_columns(plan), **summary})

            for row in convergence_rows:
                all_convergence.append({**model_columns(plan), **row})

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            status = "error"
            error = f"{type(exc).__name__}: {exc}"

            print(f"\n!!! MODEL {plan.tag} FAILED: {error}")
            traceback.print_exc()

            with open(errors_path, "a") as file:
                file.write(f"=== {plan.tag} ({plan.bundle}) ===\n")
                file.write(traceback.format_exc())
                file.write("\n")

        # Offene GIF-Auftraege abarbeiten (auch nach einem Fehler) und
        # fehlgeschlagene GIFs protokollieren.
        gif_failures = []

        if ctx is not None:

            ctx.gif_writer.close()
            gif_failures = ctx.gif_failures

            if gif_failures:
                failures_path = ctx.output_dir / "evolution_gif_failures.txt"
                failures_path.write_text("\n".join(gif_failures) + "\n")
                print(
                    f"!!! {len(gif_failures)} evolution GIFs failed, see "
                    f"{failures_path.resolve()}"
                )

        overview.append({
            **model_columns(plan),
            "attractors": "+".join(s.name for s in plan.specs),
            "seed_pixels_rgba_255": (
                "; ".join(
                    f"{a.name}={pixel_to_255(a.seed_pixel)}"
                    for a in ctx.attractors
                )
                if ctx is not None else ""
            ),
            "bundle": str(Path(plan.bundle).resolve()),
            "output_dir": str((OUTPUT_ROOT / plan.tag).resolve()),
            "status": status,
            "error": error,
            "runtime_seconds": round(time.time() - started, 1),
            "inference_steps": STEPS,
            "evolution_gif_failures": len(gif_failures),
            "evolution_replay_mismatches": (
                ctx.replay_mismatches if ctx is not None else ""
            ),
            **{
                f"convergence_{key}": value
                for key, value in model_convergence.items()
            },
        })

        # Nach jedem Modell schreiben, damit bei einem Abbruch nichts verloren geht.
        write_cross_model_outputs(
            overview, all_summaries, all_spectrum, all_convergence
        )

        del ctx

        try:
            jax.clear_caches()
        except Exception:
            pass

    print("\n\n======================================")
    print("ALL MODELS DONE")
    print("======================================")

    for row in overview:

        mean_steps = row["convergence_mean_steps_to_stable"]
        stable = (
            f"settled after {mean_steps:.1f}/{STEPS}"
            if isinstance(mean_steps, float) else "settled after ?"
        )

        mean_attractor = row["convergence_mean_steps_to_attractor"]
        attractor = (
            f"attractor after {mean_attractor:.1f}/{STEPS}"
            if isinstance(mean_attractor, float) else "attractor after ?"
        )

        print(
            f"{row['model']:28s} | {row['status']:5s} | "
            f"{row['runtime_seconds']:8.1f} s | {stable} | {attractor} | "
            f"{row['error']}"
        )

    print(f"\nTotal time: {(time.time() - total_started) / 60:.1f} min")
    print("Overview:", (OUTPUT_ROOT / "models_overview.csv").resolve())
    print("All position summaries:",
          (OUTPUT_ROOT / "all_models_position_summary.csv").resolve())
    print("All spectrum summaries:",
          (OUTPUT_ROOT / "all_models_rgb_spectrum_summary.csv").resolve())
    print("All convergence summaries:",
          (OUTPUT_ROOT / "all_models_convergence.csv").resolve())

    if errors_path.exists():
        print("Errors:", errors_path.resolve())

    print("\nDONE.")


if __name__ == "__main__":
    main()
